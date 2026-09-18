"""Audit both replications and pairwise bootstrap within independent test strata."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.repeat_early_interface import OUT,SEEDS,origin,digest
from experiments import report_early_interface as child
from experiments.capacity_probe import write_json

def main():
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    summaries={};allkeys=set()
    for seed in SEEDS:
        run=origin(seed);child.main(run);keys={r['layout_hash'] for r in json.loads((run/'games.json').read_text())}
        assert len(keys)==750 and not keys&allkeys;allkeys.update(keys);summaries[str(seed)]=json.loads((run/'summary.json').read_text())
    comparisons={}
    for size in (7,9):
        ds=[]
        for seed in SEEDS:
            rows={a:[r for r in json.loads((origin(seed)/f'{a}-games.json').read_text()) if r['size']==size and not r['automatic']] for a in ('joint','control')}
            ds.append(np.array([int(a['won'])-int(b['won']) for a,b in zip(rows['joint'],rows['control'])]))
        rng=np.random.default_rng(20261106+size);boot=[np.mean([rng.choice(d,len(d),replace=True).mean() for d in ds])*100 for _ in range(10000)]
        comparisons[str(size)]=dict(delta_pp=float(np.mean([d.mean() for d in ds])*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    summary=dict(per_seed=summaries,new_two_joint_minus_control=comparisons);write_json(OUT/'summary.json',summary)
    write_json(OUT/'independent-verification.json',dict(child_audits_pass=True,unique_layouts=len(allkeys),disjoint_tests=True,hashes_intact=True))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        for i,arm in enumerate(('joint','control')):
            cc=[summaries[str(s)][str(size)]['results'][arm] for s in SEEDS];x=np.arange(2)+(i-.5)*.32;y=[100*c['wins']/c['n'] for c in cc]
            ax.bar(x,y,width=.32,label='Adaptar encoder' if arm=='joint' else 'Encoder fijo')
            for xx,yy,c in zip(x,y,cc):ax.text(xx,yy+.3,f"{c['wins']}/{c['n']}",ha='center',fontsize=8)
        ax.set_xticks(range(2),['Semilla04','Semilla05']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.legend();ax.margins(y=.25)
    fig.savefig('research/early-interface-consistency-results.png',dpi=150);plt.close(fig)
    c=comparisons['9'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_CONSISTENCIA_INTERFAZ_TEMPRANA.md').write_text(
        '# Consistencia de adaptación a un ciclo\n\n'
        f"Principal dos nuevas9×9,adaptación−control: {c['delta_pp']:+.2f}pp,IC95%[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](early-interface-consistency-results.png)\n\n'
        'Dos lectores previamente entrenados con1ciclo,un grafo fijo.750updates adicionales prefijados porbrazo,misma inicialización/Adam/muestras dentrodepar;encoder se adapta solo en joint. '
        '1500layouts únicos de test:750porréplica,compartidos entre brazos. Bootstrapestratificadoy pareadoporlayout. '
        'Piloto anterior excluido. Ambosbrazos usan grafo;no se equiparan ciclos con tiempobiológico. Agente servido intacto.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
