"""Point 3: raw far-field density senses x variable-density training, non-lethal training, raw near senses.
See research/PROTOCOLO_DENSIDAD.md. `pilot` mode uses dev seeds and never touches reserved layouts."""
import json,pickle,shutil,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats
from minesweeper import Minesweeper
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity

OUT=Path('runs/fly-density-001');GAMES=20000;SEEDS=tuple(range(10));CONFIG=dict(rule='rpe',sparsity=.2,tuning=.35,beta=100.)
GROUPS=('fly-far-var','fly-near-var','fly-near-fixed','table-far-var','table-near-var','table-near-fixed')
LEVELS={'10':{9:8,7:5},'15':{9:12,7:7},'20':{9:16,7:10}};COUNTS={9:300,7:150}
SCHEDULE=np.random.default_rng(777).uniform(.08,.25,size=200000)   # shared by every arm: training density of game g

def training_mines(g,size,variable):return max(2,int(round(size*size*SCHEDULE[g]))) if variable else {9:12,7:7}[size]

def configure(out,games,seeds):
    global OUT,GAMES,SEEDS;OUT,GAMES,SEEDS=Path(out),games,tuple(seeds)

def build(group,seed):
    from experiments.mushroom_body import MushroomBody
    from experiments.fly_agents import RawFlyAgent,TableAgent
    from experiments.fly_senses import sniffs_raw,sniffs_far
    sense=sniffs_far if '-far-' in group else sniffs_raw
    if group.startswith('table'):return TableAgent(),None,sense
    mb=MushroomBody(**CONFIG,seed=seed);return RawFlyAgent(mb),mb,sense

def run(arm):
    from experiments.fly_agents import play
    group,seed=arm;name=f'{group}-{seed}';assert not (OUT/'completed.json').exists()
    if (OUT/f'{name}-games.json').exists():return name   # resume: finished arms are kept
    games=json.loads((OUT/'games.json').read_text());agent,mb,sense=build(group,seed)
    rng=np.random.default_rng(seed+1);eta0=mb.eta if mb is not None else agent.lr0;variable=group.endswith('-var')
    for g in range(GAMES):
        eta=eta0/(1+g/3000);size=[7,9][g%2]
        if mb is not None:mb.eta=eta
        else:agent.eta=eta
        play(agent,10_000_000+g,size,training_mines(g,size,variable),rng,True,sense=sense,lethal=False)
    rows=[dict(**g,**play(agent,g['seed'],g['size'],g['mines'],rng,False,sense=sense)) for g in games];write_json(OUT/f'{name}-games.json',rows);return name

def hand(_):
    from experiments.evaluate_fly_marker import hand as policy
    write_json(OUT/'hand-policy-games.json',policy(json.loads((OUT/'games.json').read_text())));return 'hand-policy'

def reserve_levels():
    from experiments.evaluate_difficulty_transfer import prior_layouts
    used=prior_layouts(exclude=OUT);games=[]
    for level,mines in LEVELS.items():
        for size,n in COUNTS.items():
            found=0
            for seed in range(8500000000+int(level)*1000000+size*10000,8500000000+int(level)*1000000+size*10000+9999):
                env=Minesweeper(seed,size,mines[size]);key=identity(env)
                if key in used or env.won:continue
                used.add(key);games.append(dict(seed=seed,size=size,mines=mines[size],level=level,layout_hash=key));found+=1
                if found==n:break
            assert found==n
    return games

def summarise():
    rows={p.name[:-11]:json.loads(p.read_text()) for p in OUT.glob('*-games.json')};groups=[g for g in GROUPS if f'{g}-{SEEDS[0]}' in rows];summary={}
    def interval(v):v=np.asarray(v,float);half=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return dict(mean=round(float(v.mean()),1),ci95=[round(float(v.mean()-half),1),round(float(v.mean()+half),1)],per_seed=[round(float(x),1) for x in v])
    def pct(arm,size,level):return 100*np.mean([r['won'] for r in rows[arm] if r['size']==size and (level is None or r['level']==level)])
    for size in (9,7):
        block={}
        for level in list(LEVELS)+[None]:
            per={g:[pct(f'{g}-{s}',size,level) for s in SEEDS] for g in groups};cell={g:interval(v) for g,v in per.items()};cell['hand-policy']=round(float(pct('hand-policy',size,level)),1) if 'hand-policy' in rows else None
            for a,b in (('fly-far-var','fly-near-var'),('fly-near-var','fly-near-fixed'),('fly-far-var','fly-near-fixed'),('table-far-var','table-near-var'),('table-near-var','table-near-fixed'),('fly-far-var','table-far-var'),('fly-near-fixed','table-near-fixed')):
                if a in per and b in per:d=np.array(per[a])-np.array(per[b]);t=stats.ttest_rel(per[a],per[b]);cell[f'{a} minus {b}']=dict(**interval(d),p=round(float(t.pvalue),4))
            block['all' if level is None else level]=cell
        block['unknown_step_share']={g:round(float(np.mean([sum(r['unknown_steps'] for r in rows[f'{g}-{s}'] if r['size']==size)/sum(r['steps'] for r in rows[f'{g}-{s}'] if r['size']==size) for s in SEEDS])),3) for g in groups}
        block['false_marks']={g:int(sum(r['false_marks'] for s in SEEDS for r in rows[f'{g}-{s}'] if r['size']==size)) for g in groups};summary[str(size)]=block
    write_json(OUT/'summary.json',summary);return summary

def main(pilot=False):
    global OUT,GAMES,SEEDS
    if pilot:OUT,GAMES,SEEDS=Path('benchmarks/fly-density-pilot'),5000,(100,)
    resume=not pilot and (OUT/'games.json').exists() and not (OUT/'completed.json').exists()
    if not resume:OUT.mkdir(parents=True,exist_ok=False)
    if pilot:games=[dict(seed=22_000_000+int(l)*1000+i,size=s,mines=m[s],level=l) for l,m in LEVELS.items() for s,n in ((9,100),(7,60)) for i in range(n)]
    elif resume:games=json.loads((OUT/'games.json').read_text())
    else:
        (OUT/'source').mkdir()
        for p in ['experiments/evaluate_fly_density.py','experiments/fly_agents.py','experiments/mushroom_body.py','experiments/fly_senses.py','experiments/test_fly.py','research/PROTOCOLO_DENSIDAD.md']:shutil.copy2(p,OUT/'source'/Path(p).name)
        games=reserve_levels();reserved={g['layout_hash'] for g in games}
        for variable in (False,True):assert not any(identity(Minesweeper(10_000_000+g,[7,9][g%2],training_mines(g,[7,9][g%2],variable))) in reserved for g in range(GAMES))
        with (OUT/'dataset.pkl').open('wb') as f:pickle.dump(dict(games=games),f)
        write_json(OUT/'manifest.json',dict(config=CONFIG,training='non-lethal',training_games=GAMES,seeds=list(SEEDS),groups=GROUPS,levels=LEVELS,variable_density='uniform 8-25% per game, shared schedule',unit='seed',training_layouts_in_test=0))
    if not resume:write_json(OUT/'games.json',games)
    with Pool(9,initializer=configure,initargs=(str(OUT),GAMES,SEEDS)) as pool:
        for name in pool.imap_unordered(run,[(g,s) for s in SEEDS for g in GROUPS]):print('done',name,flush=True)
        print('done',pool.apply(hand,(0,)),flush=True)
    print(json.dumps(summarise(),indent=1));write_json(OUT/'completed.json',dict(completed=True))

if __name__=='__main__':main(pilot=len(sys.argv)>1 and sys.argv[1]=='pilot')
