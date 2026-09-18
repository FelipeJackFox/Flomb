"""Independent fixed-endpoint audit with per-size paired win differences."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_nine_dagger import OUT,PARENT,MAPPING,digest,tensor_hash
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.scaled_data import identity
from experiments.capacity_probe import write_json
from experiments.diagnose_mine_choices import info_from
from minesweeper import Minesweeper


def main():
    assert json.loads((OUT/'completed.json').read_text())['completed']
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
    seal=json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    assert digest(OUT/'evaluation-checkpoints.json')==seal
    checkpoints=json.loads((OUT/'evaluation-checkpoints.json').read_text())
    parent=torch.load(PARENT,weights_only=False)
    for arm,v in checkpoints.items():
        assert digest(v['path'])==v['sha256']
        state=torch.load(v['path'],weights_only=False)
        assert tensor_hash(state['encoder'])==tensor_hash(parent['encoder'])
        if arm!='baseline':assert state['step']==2250
    games=json.loads((OUT/'games.json').read_text());collection=json.loads((OUT/'collection-games.json').read_text())
    assert len(games)==750 and len(collection)==900
    assert sum(g['size']==9 for g in games)==500 and sum(g['size']==7 for g in games)==250
    assert all(g['size']==9 and g['mines']==12 for g in collection)
    keys={g['layout_hash'] for g in games+collection}
    assert len(keys)==1650 and not keys&prior_layouts(exclude=OUT)
    for g in games+collection:assert identity(Minesweeper(g['seed'],g['size'],g['mines']))==g['layout_hash']
    for i in range(1,4):assert json.loads((OUT/f'collection-{i}.json').read_text())['teacher_actions']==0
    collected=pickle.loads((OUT/'dataset.pkl').read_bytes())['train']
    collection_keys={g['layout_hash'] for g in collection}
    for row in collected:
        assert row['layout_hash'] in collection_keys and row['size']==9 and row['mines']==12
        info=info_from(row);target=np.zeros(256,bool)
        for i in info.safe:target[(i//9)*16+i%9]=True
        assert np.array_equal(target,row['labels']) and target.any()
    expected=sorted((g['size'],g['seed'],g['layout_hash']) for g in games)
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in checkpoints}
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    assert all([r['automatic'] for r in rr]==[r['automatic'] for r in rows['baseline']] for rr in rows.values())
    summary={}
    for size in (7,9):
        stats={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']]
            wins[arm]=np.array([int(r['won']) for r in played])
            stats[arm]=dict(wins=int(wins[arm].sum()),n=len(played),
                **{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available','death_after_exact_half_min_risk')})
        comparisons={}
        for ref in ('control','baseline'):
            d=wins['dagger']-wins[ref];rng=np.random.default_rng(20261029+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            comparisons[ref]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
        summary[str(size)]=dict(results=stats,comparisons=comparisons)
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=1650,prior_overlap=0,
        checkpoints_fixed2250=True,encoder_preserved=True,hashes_intact=True,teacher_actions=0,
        public_labels_recomputed=len(collected)))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        stats=summary[str(size)]['results']
        values=[100*v['wins']/v['n'] for v in stats.values()]
        ax.bar(range(3),values,color=['#87939a','#9772bc','#0795a0'])
        for i,v in enumerate(stats.values()):ax.text(i,values[i]+.5,f"{v['wins']}/{v['n']}",ha='center')
        ax.set_xticks(range(3),['Preservado','Control2250','DAgger9 2250']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias autónomas (%)');ax.set_ylim(0,max(values)+8)
    fig.savefig('research/nine-dagger-results.png',dpi=150);plt.close(fig)
    c=summary['9']['comparisons']['control'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_DAGGER_9X9.md').write_text(
        '# DAgger con experiencia de9×9\n\n'
        f"Principal9×9, DAgger−control: {c['delta_pp']:+.2f}pp, IC95%pareado[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](nine-dagger-results.png)\n\n'
        'Se fijaron los últimos2250 updates antes del test. Los mejores por pérdida antigua son diagnósticos y no fueron evaluados aquí. '
        'Una semilla/un cerebro,500 layouts nuevos9×9 y250 nuevos7×7 compartidos entre brazos. '
        '900 partidas autónomas de colección, etiquetas públicas, cero acciones del maestro; '
        '50%datos pequeños originales en DAgger frente100%originales encontrol, mismo presupuesto de updates. '
        'Cerebro y encoder congelados; no se cambió agente servido. No confundir caída de pérdida con mejora de victorias. '
        'Intervalo bootstrap con cero eventos puede ser degenerado y no prueba tasa verdadera cero. '
        'El solver certifica solo parte de las deducciones posibles.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
