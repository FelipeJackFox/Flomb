"""Replay every visible trace to audit interventions and distinguish assisted wins."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.evaluate_assistance import OUT,MODES,digest
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.scaled_data import identity
from experiments.capacity_probe import write_json
from minesweeper import Minesweeper
from solver import analyze

def main():
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    assert json.loads((OUT/'preflight.json').read_text())['autonomous_matches_existing_evaluator']
    games=json.loads((OUT/'games.json').read_text());keys={g['layout_hash'] for g in games}
    assert len(games)==len(keys)==500 and not keys&prior_layouts(exclude=OUT)
    for g in games:assert identity(Minesweeper(g['seed'],g['size'],g['mines']))==g['layout_hash']
    expected=[g['seed'] for g in games];summary={};wins={};audited=0
    for mode in MODES:
        rows=json.loads((OUT/f'{mode}-games.json').read_text());assert [r['seed'] for r in rows]==expected
        for row,g in zip(rows,games):
            assert all(row[k]==g[k] for k in ('size','mines','layout_hash'))
            env=Minesweeper(g['seed'],g['size'],g['mines']);assert bool(env.won)==row['automatic']
            eligible=corrections=delegated=0
            for step,t in enumerate(row['trace']):
                assert env.visible.tolist()==t['visible'] and not env.done
                info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
                proposal=t['proposal'];action=t['action'];assert env.legal_mask()[proposal] and env.legal_mask()[action]
                bad=bool(info.safe and proposal not in info.safe);assert bad==t['eligible'] and info.probability_method==t['method']
                if mode=='autonomous':assert action==proposal and not t['delegated']
                elif mode in ('safe_half','safe_all'):
                    coin=int.from_bytes(hashlib.sha256(f"mosca-correction:{g['seed']}:{step}".encode()).digest()[:8],'big')<2**63
                    correct=bad and (mode=='safe_all' or coin)
                    assert bool(t['delegated'])==correct and action==(min(info.safe) if correct else proposal)
                else:
                    allowed=sorted(info.safe) if info.safe else [int(i) for i in np.flatnonzero(env.legal_mask()) if int(i) not in info.mines]
                    if not allowed:allowed=list(map(int,np.flatnonzero(env.legal_mask())))
                    assert action==min(allowed,key=lambda i:(info.mine_probability[i],i)) and t['delegated']
                eligible+=int(bad);corrections+=int(action!=proposal);delegated+=int(t['delegated']);env.step(action);audited+=1
                if env.done:assert row['death_with_safe_available']==bool(not env.won and info.safe)
            assert env.done and bool(env.won)==row['won'] and len(row['trace'])==row['clicks']
            assert (eligible,corrections,delegated)==(row['eligible'],row['corrections'],row['teacher_actions'])
        played=[r for r in rows if not r['automatic']];wins[mode]=np.array([int(r['won']) for r in played])
        summary[mode]=dict(wins=int(wins[mode].sum()),n=len(played),assisted=mode!='autonomous',
            **{k:int(sum(r[k] for r in played)) for k in ('clicks','eligible','corrections','teacher_actions','death_with_safe_available')})
        if mode in ('safe_all','public_solver'):assert summary[mode]['death_with_safe_available']==0
        if mode=='autonomous':assert summary[mode]['teacher_actions']==0
    comparisons={}
    for mode in MODES[1:]:
        d=wins[mode]-wins['autonomous'];rng=np.random.default_rng(20261102)
        boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
        comparisons[mode]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    result=dict(results=summary,assisted_minus_autonomous=comparisons)
    write_json(OUT/'summary.json',result);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=500,prior_overlap=0,trace_actions_replayed=audited,public_interventions_checked=True,hashes_intact=True))
    names=['Autónomo','Corrige50% errores seguros','Corrige100% errores seguros','Solver público completo']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    v=[100*summary[a]['wins']/summary[a]['n'] for a in MODES];ax.bar(range(4),v,color=['#099daa','#d7a03e','#d78c3e','#9c6eab'])
    for i,a in enumerate(MODES):ax.text(i,v[i]+1,f"{summary[a]['wins']}/{summary[a]['n']}",ha='center')
    ax.set_xticks(range(4),names);ax.set_ylim(0,105);ax.set_ylabel('Victorias (%)');ax.set_title('Diagnóstico asistido: solo la primera barra es autónoma')
    fig.savefig('research/assistance-diagnostic-results.png',dpi=150);plt.close(fig)
    Path('research/RESULTADO_DIAGNOSTICO_ASISTENCIA.md').write_text(
        '# Diagnóstico de correcciones públicas\\n\\n'.replace('\\n','\n')+
        '**No hubo entrenamiento. Solo el brazo autónomo mide al agente por sí mismo.** Los demás delegan decisiones en reglas externas.\n\n'
        '![Resultados](assistance-diagnostic-results.png)\n\n'
        '500tableros nuevos9×9/12compartidos,checkpoint fijo,una semilla/un cerebro. '
        'Las correcciones parciales solo intervienen cuando la propuesta ignora una segura certificada. '
        'Solver completo también decide en incertidumbre con probabilidades exactas o heurísticas según el estado. '
        'Las trayectorias divergen: diferencias no son fracciones causales aditivas ni cotas superiores matemáticas. '
        'La tasa50%se aplica a errores elegibles y se audita con una moneda fija porseed/paso. '
        'Se reconstruyeron todos los estados y acciones para comprobar que decisiones usan únicamente observaciones públicas. '
        'Agente servido ypesos intactos;estas victorias asistidas no son aprendizaje ni ventaja anatómica.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
