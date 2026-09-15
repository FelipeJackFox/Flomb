"""Audit individual runs, then bootstrap independently reserved paired tests."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.repeat_seven_rehearsal import OUT,SEEDS,origin,digest
from experiments import report_seven_rehearsal as child
from experiments.capacity_probe import write_json

def main():
    assert (OUT/'completed.json').exists()
    manifest=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in manifest['hashes'].items())
    summaries={};allkeys=set()
    for seed in SEEDS:
        run=origin(seed);child.main(run)
        keys={r['layout_hash'] for r in json.loads((run/'games.json').read_text())}
        assert len(keys)==750 and not keys&allkeys;allkeys.update(keys)
        summaries[str(seed)]=json.loads((run/'summary.json').read_text())
    comparisons={}
    for size in (7,9):
        comparisons[str(size)]={}
        for ref in ('control','baseline'):
            differences=[]
            for seed in SEEDS:
                rr={a:[r for r in json.loads((origin(seed)/f'{a}-games.json').read_text()) if r['size']==size and not r['automatic']] for a in ('replay',ref)}
                differences.append(np.array([int(a['won'])-int(b['won']) for a,b in zip(rr['replay'],rr[ref])]))
            rng=np.random.default_rng(20261101+size)
            boot=[np.mean([rng.choice(d,len(d),replace=True).mean() for d in differences])*100 for _ in range(10000)]
            comparisons[str(size)][ref]=dict(delta_pp=float(np.mean([d.mean() for d in differences])*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    summary=dict(per_seed=summaries,new_two_comparisons=comparisons)
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(child_audits_pass=True,unique_test_layouts=len(allkeys),disjoint_replica_tests=True,hashes_intact=True))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        for i,arm in enumerate(('baseline','control','replay')):
            cc=[summaries[str(s)][str(size)]['results'][arm] for s in SEEDS]
            x=np.arange(2)+(i-1)*.25;y=[100*c['wins']/c['n'] for c in cc]
            ax.bar(x,y,width=.25,label=arm)
            for xx,yy,c in zip(x,y,cc):ax.text(xx,yy+.3,f"{c['wins']}/{c['n']}",ha='center',fontsize=8)
        ax.set_xticks(range(2),['03 nueva','04 nueva']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.legend();ax.margins(y=.25)
    fig.savefig('research/seven-rehearsal-consistency-results.png',dpi=150);plt.close(fig)
    c=comparisons['7']['control'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_CONSISTENCIA_REPASO_7X7.md').write_text(
        '# Consistencia de repaso7×7\n\n'
        f"Principal dos nuevas,7×7repaso−control: {c['delta_pp']:+.2f}pp,IC95%[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](seven-rehearsal-consistency-results.png)\n\n'
        'Dos réplicas1500updates fijos desdeparent9paso2250. Mismos32ejemplos9enambosbrazos,repaso sustituye16originalespor16DAgger7heredados. '
        'Test750porréplica,1500layouts únicos nuevos en total,compartidos entre brazos dentro de réplica. '
        'Intervalos bootstrap estratificados por semilla ypareados por layout,condicionales a un cerebro. Piloto02excluido. '
        'Coste9secundario debe informarse;no inferir no-inferioridad de unintervalo inconcluso. '
        'Auditorías individuales de pesos/datos/mapeo/índices/cache/endpoint pasan. Agente servido preservado.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
