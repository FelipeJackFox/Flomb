"""Audit matched encoder adaptation, paired games and retention."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_deep_coverage import OUT,PARENT,digest,tensor_hash
from experiments.evaluate_difficulty_transfer import prior_layouts
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper
from solver import analyze

def main(out=OUT):
    global OUT,PARENT
    OUT=Path(out)
    assert (OUT/'completed.json').exists()
    m=json.loads((OUT/'manifest.json').read_text());assert all(digest(p)==sha for p,sha in m['hashes'].items())
    pair=json.loads((OUT/'pairing.json').read_text());assert pair['autonomous']==pair['assisted']
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
    collection=json.loads((OUT/'collection-games.json').read_text())
    ckeys={r['layout_hash'] for r in collection}
    assert len(collection)==len(ckeys)==400 and not ckeys&keys and not ckeys&prior_layouts(exclude=OUT)
    collection_stats=json.loads((OUT/'collection-stats.json').read_text())
    for arm in ('autonomous','assisted'):
        trajectories=json.loads((OUT/f'collection-{arm}.json').read_text())
        assert [r['layout_hash'] for r in trajectories]==[r['layout_hash'] for r in collection]
        for game,g in zip(trajectories,collection):
            env=Minesweeper(g['seed'],g['size'],g['mines']);assert identity(env)==g['layout_hash']
            for t in game['trace']:
                assert env.visible.tolist()==t['visible'] and not env.done
                info=analyze(env.observation(),env.legal_mask(),env.size,env.mine_count)
                assert env.legal_mask()[t['proposal']] and env.legal_mask()[t['action']]
                should=bool(arm=='assisted' and info.safe and t['proposal'] not in info.safe)
                assert bool(t['delegated'])==should
                assert t['action']==(min(info.safe) if should else t['proposal'])
                env.step(t['action'])
            assert env.done and bool(env.won)==game['won']
        records=pickle.loads((OUT/f'states-{arm}.pkl').read_bytes())
        assert len(records)==collection_stats[arm]['positions']
        for row in records:
            assert row['layout_hash'] in ckeys
            b=row['board'].reshape(16,16)[:9,:9].flatten()
            info=analyze(np.eye(10,dtype=np.float32)[b+1].flatten(),b==-1,9,12)
            target=np.zeros(256,bool)
            for i in info.safe:target[(i//9)*16+i%9]=True
            assert np.array_equal(target,row['labels']) and target.any()
    assert collection_stats['autonomous']['teacher_actions']==0
    expected=sorted((r['size'],r['seed'],r['layout_hash']) for r in games)
    rows={a:json.loads((OUT/f'{a}-games.json').read_text()) for a in ('baseline','autonomous','assisted')}
    for rr in rows.values():assert [(r['size'],r['seed'],r['layout_hash']) for r in rr]==expected
    assert all([r['automatic'] for r in rr]==[r['automatic'] for r in rows['baseline']] for rr in rows.values())
    summary={}
    for size in (7,9):
        counts={};wins={}
        for arm,rr in rows.items():
            played=[r for r in rr if r['size']==size and not r['automatic']];wins[arm]=np.array([int(r['won']) for r in played])
            counts[arm]=dict(wins=int(wins[arm].sum()),n=len(played),**{k:int(sum(r[k] for r in played)) for k in ('safe_choices','safe_opportunities','known_mine_choices','death_with_safe_available','death_after_exact_half_min_risk')})
        comparisons={}
        for ref in ('autonomous','baseline'):
            d=wins['assisted']-wins[ref];rng=np.random.default_rng(20261031+size)
            boot=[rng.choice(d,len(d),replace=True).mean()*100 for _ in range(10000)]
            comparisons[ref]=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())
        summary[str(size)]=dict(results=counts,comparisons=comparisons)
    write_json(OUT/'summary.json',summary);write_json(OUT/'independent-verification.json',dict(layouts_reconstructed=1150,prior_overlap=0,public_collection_replayed=True,hashes_intact=True,paired_draws=True,encoder_preserved=True,endpoint1500=True,teacher_actions_at_test=0))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,size in zip(axes,(7,9)):
        cc=list(summary[str(size)]['results'].values());values=[100*c['wins']/c['n'] for c in cc]
        ax.bar(range(3),values,color=['#899398','#9068b0','#089aa0'])
        for i,c in enumerate(cc):ax.text(i,values[i]+.4,f"{c['wins']}/{c['n']}",ha='center')
        ax.set_xticks(range(3),['Padre','Datos autónomos','Datos corregidos']);ax.set_title(f'{size}×{size}');ax.set_ylabel('Victorias (%)');ax.set_ylim(0,max(values)+7)
    fig.savefig('research/deep-coverage-results.png',dpi=150);plt.close(fig)
    c=summary['9']['comparisons']['autonomous'];lo,hi=c['ci95_pp']
    Path('research/RESULTADO_COBERTURA_PROFUNDA.md').write_text(
        '# Aprender de trayectorias con correcciones\n\n'
        f"Principal9×9, formación con correcciones−formación autónoma: {c['delta_pp']:+.2f}pp,IC95%[{lo:+.2f},{hi:+.2f}].\n\n"
        '![Resultados](deep-coverage-results.png)\n\n'
        '**Todas las partidas finales son autónomas.** El maestro solo intervino durante la recolección del brazo experimental. '
        '400layouts9 de colección compartidos,1500updatesporbrazo,750testnuevos5009/2507. '
        '16originales+16DAgger7+16experiencia9anterior+16experiencia nueva;mismo presupuesto y muestras de depósitos fijos. '
        'Trayectorias y cantidad de posiciones nuevas difieren: no se aísla profundidad como única causa. '
        'Cerebro yencoder congelados,todas las features recomputadas para elencoder actual;Adamlector/RNGheredados. '
        'Una semilla/un cerebro,último1500fijo antesdeltest,agente servido intacto.\n\n'
        'Estadísticas de colección (incluyen asistencia; NO son resultados autónomos del entrenamiento):\n```json\n'+json.dumps(collection_stats,indent=2)+'\n```\n\n'
        'Evaluación autónoma:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
