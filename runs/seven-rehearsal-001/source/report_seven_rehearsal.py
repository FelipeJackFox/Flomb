"""Audit retention and9x9 cost under matched replay substitution."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_seven_rehearsal import OUT,PARENT,digest,tensor_hash
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    pair=json.loads((OUT/'pairing.json').read_text());assert pair['control']==pair['replay']
    cache_check=json.loads((OUT/'cache-verification.json').read_text())
    assert cache_check['labels_context_masks_exact'] and cache_check['encoder_exact']
    assert digest(OUT/'evaluation-checkpoints.json')==json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    parent=torch.load(PARENT,weights_only=False)
    for arm,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items():
        assert digest(v['path'])==v['sha256'];state=torch.load(v['path'],weights_only=False)
        assert tensor_hash(state['encoder'])==tensor_hash(parent['encoder'])
        if arm!='baseline':assert state['step']==1500 and state['parent_step']==2250
    games=json.loads((OUT/'games.json').read_text());keys={r['layout_hash'] for r in games}
    assert len(games)==len(keys)==750 and not keys&prior_layouts(exclude=OUT)
    assert sum(r['size']==9 for r in games)==500 and sum(r['size']==7 for r in games)==250
    for r in games:assert identity(Minesweeper(r['seed'],r['size'],r['mines']))==r['layout_hash']
    expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games)
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ('baseline','control','replay')}
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    assert all([r['automatic'] for r in rr]==[r['automatic'] for r in rows['baseline']] for rr in rows.values())
    summary={}
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available','death_after_exact_half_min_risk')})
        comparisons={}
        for ref in ('control','baseline'):
            d=wins['replay']-wins[ref];rng=np.random.default_rng(20261031+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            comparisons[ref]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
        summary[str(size)]=dict(results=counts,comparisons=comparisons)
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=750,prior_overlap=0,hashes_intact=True,paired_draws=True,encoder_unchanged=True,endpoint1500=True))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=list(summary[str(size)]['results'].values());values=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(3),values,color=['#899398','#9068b0','#089aa0'])
        for i,c in enumerate(cc):ax.text(i,values[i]+.4,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),['Padre9','Control','Repaso7']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(values)+7)
    fig.savefig('research/seven-rehearsal-results.png',dpi=150);plt.close(fig)
    c=summary['7']['comparisons']['control'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_REPASO_7X7.md').write_text(
        '# Repaso de experiencia7×7\n\n'
        f"Principal7×7,repaso−control: {c['delta_pp']:+.2f}pp,IC95%pareado[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](seven-rehearsal-results.png)\n\n'
        'Continuación1500updates desdepadre9paso2250,últimofijoantesdetest. Misma experiencia9en32de64posiciones; '
        'repaso sustituye16de32originales por16DAgger7heredadas. Mismosíndices generados y mismos primeros16originales. '
        'Una semilla/un cerebro,250test7 y500test9nuevoscompartidos. Padre9es el baseline de esta prueba, no el modelo anterior a adaptar9. '
        'Evaluar también el coste9;no hay prueba formal de no-inferioridad. Intervalos condicionales bootstrap pareado. '
        'Sin recolecciónnueva ni accionesdelmaestro,encoder/grafo fijos,agenteservido intacto.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
