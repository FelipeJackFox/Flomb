"""Audit and paired comparison: learned synapses versus untouched brain and versus more reader training."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_synaptic_learning import OUT,ARMS,UPDATES,BASE,digest
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

LABELS=dict(baseline='Cerebro intacto',reader='Solo aprende el lector',brain='Solo aprenden las sinapsis')
CONTRASTS=(('brain','baseline'),('brain','reader'),('reader','baseline'))

def main(out=OUT):
    global OUT
    OUT=Path(out)
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    assert digest(OUT/'evaluation-checkpoints.json')==json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    saved={a:torch.load(v['path'],weights_only=False) for a,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items() if digest(v['path'])==v['sha256']}
    assert set(saved)=={'warmup','reader','brain'} and saved['brain']['step']==UPDATES==saved['reader']['step']
    assert saved['brain']['draw_blocks']==saved['reader']['draw_blocks']
    original=torch.load(BASE/'training/retina_plastic-20260926.pt',weights_only=False)['model']
    delta=(saved['brain']['edge_log_gain']-original['edge_log_gain']);node=(saved['brain']['node_log_gain']-original['node_log_gain'])
    synapses=dict(edges_total=int(delta.numel()),edges_modified=int((delta!=0).sum()),edges_changed_over_1pct=int((delta.abs()>.01).sum()),edges_changed_over_10pct=int((delta.abs()>.0953).sum()),
        mean_abs_log_change_of_modified=float(delta[delta!=0].abs().mean()) if (delta!=0).any() else 0.,neurons_gain_modified=int((node!=0).sum()))
    games=json.loads((OUT/'games.json').read_text());keys={g['layout_hash'] for g in games}
    assert len(games)==len(keys)==750 and not keys&prior_layouts(exclude=OUT)
    assert sum(g['size']==9 for g in games)==500 and sum(g['size']==7 for g in games)==250
    for g in games:assert identity(Minesweeper(g['seed'],g['size'],g['mines']))==g['layout_hash']
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ARMS}
    expected=sorted((g['size'],g['seed'],g['layout_hash']) for g in games)
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    for a in ARMS[1:]:assert [r['automatic'] for r in rows[a]]==[r['automatic'] for r in rows['baseline']]
    summary=dict(synapses=synapses,history=json.loads((OUT/'history-brain.json').read_text()))
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available')})
        summary[str(size)]=dict(results=counts)
        for a,b in CONTRASTS:
            d=wins[a]-wins[b];rng=np.random.default_rng(20261111+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            summary[str(size)][f'{a}_minus_{b}']=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=750,prior_overlap=0,hashes_intact=True,identical_positions_reader_and_brain=True,endpoint=UPDATES))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=[summary[str(size)]['results'][a] for a in ARMS];v=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(3),v,color=['#8a8f98','#099caa','#aa7945'])
        for i,c in enumerate(cc):ax.text(i,v[i]+.5,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=8);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(v)+7)
    pilot=OUT.name=='synaptic-learning-001'
    fig.savefig('research/synaptic-learning-results.png' if pilot else OUT/'results.png',dpi=150);plt.close(fig)
    c=summary['9']['brain_minus_baseline'];lo,hi=c['ci95_pp']
    report=Path('research/RESULTADO_APRENDIZAJE_SINAPTICO.md') if pilot else OUT/'report.md'
    figure='synaptic-learning-results.png' if pilot else 'results.png'
    report.write_text(
        '# ¿Aprende el cerebro?\n\n'
        f"Principal 9×9, sinapsis entrenadas − cerebro intacto: {c['delta_pp']:+.2f} pp, IC95% [{lo:+.2f}, {hi:+.2f}].\n\n"
        f'![Resultados]({figure})\n\n'
        f"Tres ciclos de propagación. Tras un calentamiento común del lector (3000 updates sobre el cerebro intacto), el brazo de sinapsis congela encoder y lector y entrena solo "
        f"ganancias por conexión y por neurona durante {UPDATES} updates; el brazo lector hace lo contrario con exactamente las mismas posiciones. "
        f"Conexiones modificadas: {synapses['edges_modified']:,} de {synapses['edges_total']:,}; con cambio mayor a 10%: {synapses['edges_changed_over_10pct']:,}. "
        'Topología y signos fijos; modelo de tasas sin spikes. Una inferencia por jugada, 750 layouts nuevos (500 de 9×9, 250 de 7×7), endpoint fijo sin selección por test. '
        'Una semilla y un cerebro: un resultado favorable exige réplica y el control de grafo recableado antes de atribuirlo a la anatomía.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
