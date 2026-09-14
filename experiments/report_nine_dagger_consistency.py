"""Audit new replicates and paired performance by board size."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.repeat_nine_dagger import OUT,SEEDS,origin,parent,digest,tensor_hash
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from experiments.diagnose_mine_choices import info_from
from minesweeper import Minesweeper


def main():
    assert (OUT/'completed.json').exists()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
    seal=json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    assert digest(OUT/'evaluation-checkpoints.json')==seal
    for key,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items():
        assert digest(v['path'])==v['sha256']
        seed,arm=key.split('-');state=torch.load(v['path'],weights_only=False)
        baseline=torch.load(parent(int(seed)),weights_only=False)
        assert tensor_hash(state['encoder'])==tensor_hash(baseline['encoder'])
        if arm!='baseline':assert state['step']==2250
    games=json.loads((OUT/'games.json').read_text());collection=json.loads((OUT/'collection-games.json').read_text())
    keys={r['layout_hash'] for r in games+collection};assert len(keys)==1650 and len(games)==750 and len(collection)==900
    assert sum(r['size']==9 for r in games)==500 and sum(r['size']==7 for r in games)==250
    assert all(r['size']==9 and r['mines']==12 for r in collection)
    for r in games+collection:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
    allowed={OUT,*[origin(s) for s in SEEDS[1:]]};used=set()
    for p in Path('runs').glob('*/dataset.pkl'):
        if p.parent in allowed:continue
        for rr in pickle.loads(p.read_bytes()).values():
            if isinstance(rr,list):used.update(r['layout_hash'] for r in rr if isinstance(r,dict) and 'layout_hash' in r)
    for p in Path('runs').glob('*/*games.json'):
        if p.parent in allowed:continue
        rr=json.loads(p.read_text())
        if isinstance(rr,list):used.update(r['layout_hash'] for r in rr if isinstance(r,dict) and 'layout_hash' in r)
    assert not keys&used
    nlabels={}
    for seed in SEEDS[1:]:
        run=origin(seed);assert json.loads((run/'collection-games.json').read_text())==collection
        assert json.loads((run/'games.json').read_text())==games
        child=json.loads((run/'manifest.json').read_text())
        assert all(digest(p)==sha for p,sha in child['hashes'].items())
        for i in range(1,4):assert json.loads((run/f'collection-{i}.json').read_text())['teacher_actions']==0
        records=pickle.loads((run/'dataset.pkl').read_bytes())['train'];nlabels[str(seed)]=len(records)
        collection_keys={r['layout_hash'] for r in collection}
        for r in records:
            assert r['layout_hash'] in collection_keys and r['size']==9
            info=info_from(r);labels=np.zeros(256,bool)
            for i in info.safe:labels[(i//9)*16+i%9]=True
            assert np.array_equal(labels,r['labels']) and labels.any()
    expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games)
    allrows={};auto=None
    for seed in SEEDS:
        for arm in ('baseline','control','dagger'):
            rr=json.loads((OUT/f'{seed}-{arm}-games.json').read_text())
            assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
            flags=[r['automatic'] for r in rr]
            if auto is None:auto=flags
            assert flags==auto;allrows[(seed,arm)]=rr
    summary={}
    for size in (7,9):
        counts={};wins={}
        for (seed,arm),rr in allrows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']]
            w=np.array([int(r['won']) for r in played]);wins[(seed,arm)]=w
            counts[f'{seed}-{arm}']=dict(wins=int(w.sum()),n=len(w),**{m:int(sum(r[m] for r in played)) for m in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available','death_after_exact_half_min_risk')})
        comparisons={}
        for group,seeds in [('new_two',SEEDS[1:]),('all_three',SEEDS)]:
            comparisons[group]={}
            for ref in ('control','baseline'):
                d=np.array([wins[(s,'dagger')]-wins[(s,ref)] for s in seeds]).mean(axis=0)
                rng=np.random.default_rng(20261030+size)
                boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
                comparisons[group][ref]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
        summary[str(size)]=dict(results=counts,comparisons=comparisons)
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=1650,prior_overlap=0,
        public_labels_recomputed=nlabels,hashes_intact=True,endpoint2250=True,encoder_frozen=True,teacher_actions=0))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        for i,arm in enumerate(('baseline','control','dagger')):
            cc=[summary[str(size)]['results'][f'{s}-{arm}'] for s in SEEDS]
            xx=np.arange(3)+(i-1)*.25;vv=[100*c['wins']/c['n'] for c in cc]
            ax.bar(xx,vv,width=.25,label=arm)
            for x,y,c in zip(xx,vv,cc):ax.text(x,y+.3,str(c['wins']),ha='center',fontsize=8)
        ax.set_xticks(range(3),['02 reutilizada','03 nueva','04 nueva']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.legend();ax.margins(y=.2)
    fig.savefig('research/nine-dagger-consistency-results.png',dpi=150);plt.close(fig)
    c=summary['9']['comparisons']['new_two']['control'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_CONSISTENCIA_DAGGER_9X9.md').write_text(
        '# Consistencia DAgger9×9\n\n'
        f"Principal dos semillas nuevas,9×9,DAgger−control: {c['delta_pp']:+.2f}pp, IC95%pareado[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](nine-dagger-consistency-results.png)\n\n'
        'Cada barra etiqueta victorias absolutas; denominadores500en9 y250en7 salvo aperturasautomáticas excluidas detalladas abajo. '
        'Dos nuevos pares2250updates;piloto02solo reevaluado. Últimos2250fijados antesdeltest, sin selección por partidas. '
        '900layouts de colección compartidos entre dos nuevos con trayectorias propias;750test nuevos compartidos por tres políticas. '
        'Bootstrap remuestrea layouts conjuntamente, no cerebros. Con pocos éxitos puede ser degenerado. '
        'Cerebro/encodercongelados,maestroetiqueta pero no actúa, agente servido sin cambios. '
        'La comparación de retención7 es secundaria y no debe ocultarse tras mejora9.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
