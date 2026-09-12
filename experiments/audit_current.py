"""Read-only paired audit of immutable policy checkpoints; writes audit outputs only."""
import argparse, json, hashlib, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import sparse
from brain import BrainPolicy
from minesweeper import Minesweeper
from curriculum import encode,decode
from train_curriculum import STRATA


def wilson(w,n):
    if not n:return [0.,1.]
    z=1.95996398454;p=w/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-radius),min(1.,mid+radius)]


def history(run):
    rows={}
    for path in sorted(run.glob('metrics-from-*.jsonl')):
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            rows[r['episode']]=r # later resumed log supersedes replayed tail
    buckets=defaultdict(list)
    for episode,r in sorted(rows.items()):
        window=(episode-1)//8000
        buckets[(window,r['size'],r['mines'])].append(r)
    output=[]
    for (window,size,mines),rr in sorted(buckets.items()):
        decisions=[r for r in rr if not r['automatic_win']]
        wins=sum(r['won'] for r in decisions);n=len(decisions)
        output.append(dict(window_start=window*8000+1,window_end=(window+1)*8000,size=size,mines=mines,episodes=len(rr),automatic_wins=len(rr)-n,decision_wins=wins,decision_episodes=n,decision_rate=wins/n if n else None,wilson95=wilson(wins,n)))
    return {'latest_episode':max(rows,default=0),'unique_episodes':len(rows),'exact_strata':output}


def audit(run, count=32, workers=2, checkpoint=None):
    state=json.loads((run/'state.json').read_text())
    checkpoint=checkpoint or run/state['checkpoint']
    paths={'current':checkpoint,'initial':run/'initial-expanded.npz','mid_16384':run/'checkpoint-016384.npz'}
    paths={k:p for k,p in paths.items() if p.exists()}
    out={'checkpoint_state_snapshot':state,'count_per_stratum':count,'seed_start':2100000000,'protocol':'stochastic policy; independent action RNG initialized seed+10000 for each policy on each common board; no teacher; first safe center automatic', 'checkpoints':{k:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},'history':history(run),'results':{},'paired':{}}
    graph=sparse.load_npz('data/processed/graph.npz');policy=BrainPolicy(graph,inputs=2560,actions=256,sparse_workers=workers)
    try:
        for name,path in list(paths.items())+[('random',None)]:
            if path:policy.load(path)
            records={}
            for group,(size,mines) in enumerate(STRATA):
                key=f'{size}x{size}-{mines}';games=[]
                for j in range(count):
                    seed=2100000000+group*100000+j;env=Minesweeper(seed,size,mines);rng=np.random.default_rng(seed+10000)
                    automatic=bool(env.won);clicks=0
                    while not env.done:
                        if path:
                            obs,mask=encode(env);probs,_=policy.forward(obs,mask,cache=False)
                            action=decode(rng.choice(len(probs),p=probs),size)
                        else:action=int(rng.choice(np.flatnonzero(env.legal_mask())))
                        env.step(action);clicks+=1
                    games.append({'seed':seed,'won':bool(env.won),'automatic_win':automatic,'clicks':clicks,'safe_fraction':float(np.sum(env.visible>=0)/(size*size-mines))})
                decisions=[g for g in games if not g['automatic_win']];w=sum(g['won'] for g in decisions);n=len(decisions)
                records[key]={'games':games,'wins':sum(g['won'] for g in games),'automatic_wins':count-n,'decision_wins':w,'decision_episodes':n,'decision_win_rate':w/n if n else None,'wilson95':wilson(w,n),'mean_safe_fraction':float(np.mean([g['safe_fraction'] for g in games]))}
                print(json.dumps({'model':name,'stratum':key,**{k:v for k,v in records[key].items() if k!='games'}}),flush=True)
            out['results'][name]=records
        for other in out['results']:
            if other=='current':continue
            paired={}
            for key,current in out['results']['current'].items():
                games=zip(current['games'],out['results'][other][key]['games'])
                better=worse=0;diff=[]
                for a,b in games:
                    better+=a['won'] and not b['won'];worse+=b['won'] and not a['won'];diff.append(a['safe_fraction']-b['safe_fraction'])
                discordant=better+worse
                p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(better,worse)+1))/2**discordant) if discordant else 1.
                paired[key]={'current_only_wins':int(better),'other_only_wins':int(worse),'mcnemar_exact_two_sided_p':p,'mean_safe_fraction_delta':float(np.mean(diff))}
            out['paired'][other]=paired
    finally:policy.close()
    return out


def report(out,path):
    lines=['# Auditoría de la corrida actual','',f"Checkpoint fijado: `{out['checkpoints']['current']['path']}`; SHA256 `{out['checkpoints']['current']['sha256']}`.",'',f"{out['count_per_stratum']} tableros nuevos por estrato, idénticos para actual, inicial, intermedio y aleatorio. Políticas estocásticas, sin solver. Semillas desde 2,100,000,000. Intervalos Wilson95 excluyen victorias automáticas. No se detuvo ni modificó entrenamiento.",'','## Evaluación emparejada','','| Estrato | Modelo | Victorias no automáticas | Wilson95 | Fracción segura media |','|---|---|---:|---|---:|']
    for key in next(iter(out['results'].values())):
        for name,rr in out['results'].items():
            r=rr[key];lo,hi=r['wilson95'];lines.append(f"| {key} | {name} | {r['decision_wins']}/{r['decision_episodes']} | {lo:.1%}–{hi:.1%} | {r['mean_safe_fraction']:.3f} |")
    lines+=['','## Discordancias emparejadas','','| Comparador | Estrato | Solo actual gana | Solo comparador gana | p exacta McNemar | Δ fracción segura |','|---|---|---:|---:|---:|---:|']
    for name,rr in out['paired'].items():
        for key,r in rr.items():lines.append(f"| {name} | {key} | {r['current_only_wins']} | {r['other_only_wins']} | {r['mcnemar_exact_two_sided_p']:.3f} | {r['mean_safe_fraction_delta']:+.3f} |")
    lines+=['','## Histórico por dificultad exacta','','Se deduplicaron episodios repetidos tras reanudación, priorizando el log más reciente. Las ventanas son de 8,000 episodios globales; tamaño y número de minas exactos, sin mezclar currículo. Las filas pequeñas tienen incertidumbre amplia. Los históricos son on-policy y no sustituyen la evaluación emparejada.','','| Ventana | Tablero/minas | Victorias decisiones | Automáticas | Wilson95 |','|---|---|---:|---:|---|']
    for r in out['history']['exact_strata']:
        lo,hi=r['wilson95'];lines.append(f"| {r['window_start']}–{r['window_end']} | {r['size']}x{r['size']}/{r['mines']} | {r['decision_wins']}/{r['decision_episodes']} | {r['automatic_wins']} | {lo:.1%}–{hi:.1%} |")
    lines+=['',f"Histórico hasta episodio {out['history']['latest_episode']}; última ventana parcial. El inicial es el piloto de 2,176 episodios expandido a 16×16, no un modelo sin entrenamiento.",'','## Interpretación','','Una auditoría de 32 tableros por estrato es preliminar: cero victorias no implica capacidad exactamente cero. Las pruebas por estrato son exploratorias, sin corrección por comparaciones múltiples. Cambios en fracción revelada pueden ser más sensibles que victorias, pero no prueban estrategias útiles. No afirmar mejora si victorias emparejadas e intervalos no la respaldan.']
    path.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=Path('runs/curriculum-001'));ap.add_argument('--count',type=int,default=32);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--checkpoint',type=Path);args=ap.parse_args()
    result=audit(args.run,args.count,args.workers,args.checkpoint)
    dest=Path('runs/audits');dest.mkdir(exist_ok=True);path=dest/f'current-{int(time.time())}.json';path.write_text(json.dumps(result,indent=2));report(result,Path('research/AUDITORIA_CORRIDA_ACTUAL.md'));print('AUDIT_OUTPUT',path,flush=True)
