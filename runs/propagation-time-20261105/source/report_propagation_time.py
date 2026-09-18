"""Matched propagation-time audit and autonomous game comparison."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.compare_propagation_time import OUT,digest
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main(out=OUT):
    global OUT
    OUT=Path(out)
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    pairs=json.loads((OUT/'pairing.json').read_text());assert pairs['early']==pairs['late']
    input_check=json.loads((OUT/'input-verification.json').read_text())
    assert input_check['labels_masks_context_exact'] and input_check['one_and_three_nonzero_distinct']
    assert digest(OUT/'evaluation-checkpoints.json')==json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    for arm,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items():
        assert digest(v['path'])==v['sha256'];state=torch.load(v['path'],weights_only=False)
        assert state['step']==3000 and state['arm']==arm
        assert sum(t.numel() for t in state['head'].values())==12769
    games=json.loads((OUT/'games.json').read_text());keys={g['layout_hash'] for g in games}
    assert len(games)==len(keys)==750 and not keys&prior_layouts(exclude=OUT)
    assert sum(g['size']==9 for g in games)==500 and sum(g['size']==7 for g in games)==250
    for g in games:assert identity(Minesweeper(g['seed'],g['size'],g['mines']))==g['layout_hash']
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ('early','late')}
    expected=sorted((g['size'],g['seed'],g['layout_hash']) for g in games)
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    assert [r['automatic'] for r in rows['early']]==[r['automatic'] for r in rows['late']]
    summary={}
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available')})
        d=wins['early']-wins['late'];rng=np.random.default_rng(20261104+size)
        boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
        summary[str(size)]=dict(results=counts,early_minus_late=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist()))
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=750,prior_overlap=0,hashes_intact=True,paired_initialization_and_samples=True,parameters=12769,endpoint3000=True))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=list(summary[str(size)]['results'].values());v=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(2),v,color=['#aa7945','#099caa'])
        for i,c in enumerate(cc):ax.text(i,v[i]+.5,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(2),['Actividad tras1 ciclo','Actividad tras3 ciclos']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(v)+7)
    pilot=OUT.name=='propagation-time-001'
    fig.savefig('research/propagation-time-results.png' if pilot else OUT/'results.png',dpi=150);plt.close(fig)
    c=summary['9']['early_minus_late'];lo,hi=c['ci95_pp']
    report=Path('research/RESULTADO_TIEMPO_PROPAGACION.md') if pilot else OUT/'report.md'
    figure='propagation-time-results.png' if pilot else 'results.png'
    report.write_text(
        '# Lectura temprana frente a tardía\n\n'
        f"Principal9×9,1ciclo−3ciclos: {c['delta_pp']:+.2f}pp,IC95%[{lo:+.2f},{hi:+.2f}].\n\n"
        f'![Resultados]({figure})\n\n'
        'Ambos usan el grafo cerebral y las mismas10clases de salida,encoder/gains/pesos fijos. '
        'Solo varían pasos de propagación antes de extraer actividad:1vs3. '
        'Lectores idénticos12769parámetros,misma inicialización nueva,Adam,3000updates ybatches. '
        'Encoder fue adaptado históricamente a3ciclos;esta asimetría limita la interpretación. '
        '750layouts nuevos compartidos5009/2507,una inicialización/un cerebro. '
        'No se equiparan ciclosdiscretos con tiempo biológico ni se infiere capacidad máxima. '
        'No entran pistas crudas directamente allector. Agente servido intacto.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
