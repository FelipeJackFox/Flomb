"""Audit and paired comparison of learned orientation fusion against the logit mean."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_orientation_fusion import OUT,ARMS,PARAMETERS,digest
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

LABELS=dict(mean='Promedio de logits',fusion='Fusión aprendida',fusion_aug='Fusión + giros')

def main(out=OUT):
    global OUT
    OUT=Path(out)
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    pairs=json.loads((OUT/'pairing.json').read_text());assert len({p['draws'] for p in pairs.values()})==1 and pairs['fusion']==pairs['fusion_aug']
    input_check=json.loads((OUT/'input-verification.json').read_text())
    assert input_check['labels_masks_context_exact'] and input_check['aligned_rotations']
    assert digest(OUT/'evaluation-checkpoints.json')==json.loads((OUT/'evaluation-seal.json').read_text())['sha256']
    for arm,v in json.loads((OUT/'evaluation-checkpoints.json').read_text()).items():
        assert digest(v['path'])==v['sha256'];state=torch.load(v['path'],weights_only=False)
        assert state['step']==3000 and state['arm']==arm
        assert sum(t.numel() for t in state['head'].values())==PARAMETERS[arm]
    games=json.loads((OUT/'games.json').read_text());keys={g['layout_hash'] for g in games}
    assert len(games)==len(keys)==750 and not keys&prior_layouts(exclude=OUT)
    assert sum(g['size']==9 for g in games)==500 and sum(g['size']==7 for g in games)==250
    for g in games:assert identity(Minesweeper(g['seed'],g['size'],g['mines']))==g['layout_hash']
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ARMS}
    expected=sorted((g['size'],g['seed'],g['layout_hash']) for g in games)
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    for a in ARMS[1:]:assert [r['automatic'] for r in rows[a]]==[r['automatic'] for r in rows['mean']]
    summary={}
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available')})
        summary[str(size)]=dict(results=counts)
        for arm in ARMS[1:]:
            d=wins[arm]-wins['mean'];rng=np.random.default_rng(20261110+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            summary[str(size)][f'{arm}_minus_mean']=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=750,prior_overlap=0,hashes_intact=True,paired_samples=True,parameters=PARAMETERS,endpoint3000=True))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=[summary[str(size)]['results'][a] for a in ARMS];v=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(3),v,color=['#099caa','#aa7945','#7a4fa3'])
        for i,c in enumerate(cc):ax.text(i,v[i]+.5,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS]);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(v)+7)
    pilot=OUT.name=='orientation-fusion-001'
    fig.savefig('research/orientation-fusion-results.png' if pilot else OUT/'results.png',dpi=150);plt.close(fig)
    c=summary['9']['fusion_aug_minus_mean'];lo,hi=c['ci95_pp']
    report=Path('research/RESULTADO_FUSION_ORIENTACIONES.md') if pilot else OUT/'report.md'
    figure='orientation-fusion-results.png' if pilot else 'results.png'
    report.write_text(
        '# Fusión aprendida de orientaciones\n\n'
        f"Principal 9×9, fusión con giros − promedio de logits: {c['delta_pp']:+.2f} pp, IC95% [{lo:+.2f}, {hi:+.2f}].\n\n"
        f'![Resultados]({figure})\n\n'
        'Los tres brazos usan cuatro pasadas del cerebro por jugada, el mismo encoder y grafo fijos con 1 ciclo, lectores nuevos, '
        'Adam .001, 3000 updates y los mismos draws de posiciones. El promedio entrena un lector de una orientación (12769 parámetros) '
        'y promedia logits girados de vuelta; la fusión lee las cuatro actividades alineadas a la vez (12739 parámetros). '
        '750 layouts nuevos 500 de 9×9 y 250 de 7×7. Sin selección de endpoint por test. No demuestra ventaja anatómica.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
