"""Audit matched encoder adaptation, paired games and retention."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_safe_mass import OUT,PARENT,digest,tensor_hash
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main(out=OUT):
    global OUT,PARENT
    OUT=Path(out)
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    pair=json.loads((OUT/'pairing.json').read_text());assert pair['uniform']==pair['mass']
    assert digest(OUT/'evaluation-checkpoints.json')==json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    PARENT=Path(json.loads((OUT/'evaluation-checkpoints.json').read_text())['baseline']['path'])
    parent=torch.load(PARENT,weights_only=False)
    for arm,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items():
        assert digest(v['path'])==v['sha256'];state=torch.load(v['path'],weights_only=False)
        assert tensor_hash(state['encoder'])==tensor_hash(parent['encoder'])
        if arm!='baseline':assert state['step']==1500
    games=json.loads((OUT/'games.json').read_text());keys={r['layout_hash'] for r in games}
    assert len(games)==len(keys)==750 and not keys&prior_layouts(exclude=OUT)
    assert sum(r['size']==9 for r in games)==500 and sum(r['size']==7 for r in games)==250
    for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
    expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games)
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ('baseline','uniform','mass')}
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    assert all([r['automatic'] for r in rr]==[r['automatic'] for r in rows['baseline']] for rr in rows.values())
    summary={}
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available','death_after_exact_half_min_risk')})
        comparisons={}
        for ref in ('uniform','baseline'):
            d=wins['mass']-wins[ref];rng=np.random.default_rng(20261031+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            comparisons[ref]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
        summary[str(size)]=dict(results=counts,comparisons=comparisons)
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=750,prior_overlap=0,hashes_intact=True,paired_draws=True,encoder_preserved=True,endpoint1500=True))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=list(summary[str(size)]['results'].values());values=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(3),values,color=['#899398','#9068b0','#089aa0'])
        for i,c in enumerate(cc):ax.text(i,values[i]+.4,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),['Padre','CE uniforme','Masa segura']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(values)+7)
    fig.savefig('research/safe-mass-results.png',dpi=150);plt.close(fig)
    c=summary['9']['comparisons']['uniform'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_MASA_SEGURA.md').write_text(
        '# Probabilidad total de elegir una segura\n\n'
        f"Principal9×9, masa segura−CEuniforme: {c['delta_pp']:+.2f}pp,IC95%[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](safe-mass-results.png)\n\n'
        '1500updates fijos desdeparent, mismos pesos/Adam/RNG/índices ybatches. '
        'Solo cambia objetivo: CEuniforme sobretodaslasetiquetasseguras frente−logdelaprobabilidadtotaldelconjuntoseguro. '
        'No hay filtros delmaestro eninferencia,etiquetas usan solo información pública. '
        'Encoder/grafo fijos,features recomputadas paraencoderactual,datosidénticos16original+16old7+16old9+16new9autónomos. '
        'Test750layouts nuevos compartidos5009/2507,una semilla/un cerebro. '
        'Último1500prefijado;no comparar pérdidas numéricas entre objetivos como evidencia de rendimiento. '
        'Retención7secundaria,agente servido intacto.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
