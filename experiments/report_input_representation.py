"""Audit and paired analysis of the predeclared input comparison."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.compare_input_representation import OUT, SEEDS, digest
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper

def main():
    assert (OUT/'completed.json').exists()
    manifest = json.loads((OUT/'manifest.json').read_text())
    games = json.loads((OUT/'games.json').read_text())
    keys = {r['layout_hash'] for r in games}; assert len(keys) == 500
    for r in games:
        assert identity(Minesweeper(r['seed'], r['size'], r['mines'])) == r['layout_hash']
    used = set()
    for path in Path('runs').glob('*/dataset.pkl'):
        if path.parent == OUT: continue
        for rows in pickle.loads(path.read_bytes()).values():
            if isinstance(rows, list):
                used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    for path in Path('runs').glob('*/*games.json'):
        if path.parent == OUT: continue
        rows = json.loads(path.read_text())
        if isinstance(rows, list):
            used.update(r['layout_hash'] for r in rows if isinstance(r, dict) and 'layout_hash' in r)
    assert not keys & used
    assert all(digest(p) == sha for p, sha in manifest['hashes'].items())
    assert digest(OUT/'selection.json') == json.loads((OUT/'selection-seal.json').read_text())['sha256']
    pairing = json.loads((OUT/'pairing.json').read_text())
    selected = json.loads((OUT/'selection.json').read_text())
    results = json.loads((OUT/'results.json').read_text())
    wins = {}; expected = sorted((r['size'], r['seed'], r['layout_hash']) for r in games); reference = None
    for seed in SEEDS:
        assert pairing[f'raw-{seed}'] == pairing[f'brain-{seed}']
        for variant in ('raw', 'brain'):
            key = f'{variant}-{seed}'
            hist = json.loads((OUT/f'history-{key}.json').read_text())
            assert [r['step'] for r in hist] == list(range(0, 5001, 250))
            chosen = min(hist, key=lambda r: r['validation_loss'])
            assert selected[key] == dict(step=chosen['step'], validation_loss=chosen['validation_loss'])
            saved = torch.load(OUT/f'best-{key}.pt', weights_only=False)
            assert saved['step'] == chosen['step']
            rows = json.loads((OUT/f'{key}-games.json').read_text())
            assert [(r['size'], r['seed'], r['layout_hash']) for r in rows] == expected
            auto = [r['automatic'] for r in rows]
            if reference is None: reference = auto
            assert auto == reference
            played = [r for r in rows if not r['automatic']]
            wins[key] = np.array([int(r['won']) for r in played])
            assert int(wins[key].sum()) == results[key]['wins'] and len(played) == results[key]['n']
            for metric in ('known_mine_choices', 'safe_choices', 'safe_opportunities'):
                assert sum(r[metric] for r in played) == results[key][metric]
    delta = np.stack([wins[f'raw-{s}']-wins[f'brain-{s}'] for s in SEEDS]).mean(0)
    rng = np.random.default_rng(20261018)
    boot = [rng.choice(delta, len(delta), replace=True).mean()*100 for _ in range(10000)]
    summary = dict(raw_mean_pct=float(np.mean([wins[f'raw-{s}'].mean() for s in SEEDS])*100),
                   brain_mean_pct=float(np.mean([wins[f'brain-{s}'].mean() for s in SEEDS])*100),
                   raw_minus_brain_pp=float(delta.mean()*100), conditional_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
                   n_shared_layouts=500, n_played=len(delta), automatic_excluded=sum(reference))
    write_json(OUT/'summary.json', summary)
    write_json(OUT/'independent_verification.json', dict(layouts_reconstructed=500, prior_overlap=0,
               paired_initialization_batches=True, selection_recomputed=True, seal_intact=True,
               result_counts_verified=True, original_hashes_unchanged=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    colors = {'raw':'#098c99', 'brain':'#a967ae'}
    labels = {'raw':'Pistas visibles', 'brain':'Actividad cerebral'}
    for j, variant in enumerate(('raw', 'brain')):
        vals = [100*wins[f'{variant}-{s}'].mean() for s in SEEDS]; x = np.arange(3)+(j-.5)*.34
        axes[0].bar(x, vals, .34, label=labels[variant], color=colors[variant])
        for xx, val, seed in zip(x, vals, SEEDS):
            axes[0].text(xx, val+1, str(results[f'{variant}-{seed}']['wins']), ha='center')
        for i, seed in enumerate(SEEDS):
            hist = json.loads((OUT/f'history-{variant}-{seed}.json').read_text())
            axes[1].plot([r['step'] for r in hist], [r['validation_loss'] for r in hist],
                         color=colors[variant], alpha=.65, label=labels[variant] if i==0 else None)
    axes[0].set_xticks(np.arange(3), [str(s) for s in SEEDS]); axes[0].set_ylabel('Victorias autónomas (%)')
    axes[0].set_ylim(0, min(100, max(100*v.mean() for v in wins.values())+15))
    axes[0].set_title(f"7×7 / 7 minas · {len(delta)} partidas con decisiones"); axes[0].legend()
    axes[1].set(xlabel='Actualizaciones', ylabel='Pérdida de validación', title='Tres semillas por entrada'); axes[1].legend()
    fig.savefig('research/input-representation-results.png', dpi=150); plt.close(fig)
    table = '\n'.join(f"| {s} | {results[f'raw-{s}']['wins']}/{len(delta)} | {results[f'brain-{s}']['wins']}/{len(delta)} | {selected[f'raw-{s}']['step']} / {selected[f'brain-{s}']['step']} |" for s in SEEDS)
    lo, hi = summary['conditional_ci95_pp']
    doc = f'''# Comparación controlada: pistas visibles y actividad cerebral

Pistas visibles: **{summary['raw_mean_pct']:.2f}%** de victorias autónomas; actividad cerebral: **{summary['brain_mean_pct']:.2f}%**. Diferencia pistas menos cerebro **{summary['raw_minus_brain_pp']:+.2f} puntos porcentuales**, IC95% condicional [{lo:+.2f}, {hi:+.2f}].

| Semilla | Pistas visibles | Actividad cerebral | Updates elegidos: pistas / cerebro |
|---|---|---|---|
{table}

![Resultados y pérdidas](input-representation-results.png)

Ambos lectores tienen 12,769 parámetros y reciben diez canales espaciales más tamaño/número de minas públicos. Inicialización y minibatches idénticos dentro de cada par, comprobados por hash. Cada lector se entrenó desde cero durante 5,000 actualizaciones Adam sobre las mismas 10,000 posiciones originales. Se eligió mínima pérdida en el holdout histórico de 1,000 posiciones (paso0 y cada250) antes de evaluar. No hay nuevos datos DAgger en esta comparación.

Los 500 layouts finales 7×7/7 minas son nuevos, compartidos por los seis lectores y disjuntos de datasets/listas de juegos previos. Hay {sum(reference)} aperturas ganadoras automáticas excluidas. Los intervalos usan 10,000 remuestreos por tablero y promedian las diferencias entre las tres semillas dentro del tablero. Son condicionales a estos lectores y un único cerebro previamente entrenado; no son 1,500 tableros independientes ni tres cerebros.

## Alcance

Esta prueba mide la facilidad de uso de dos representaciones por este lector y presupuesto. Una ventaja de pistas directas implica un coste de la representación cerebral actual, que incluye encoder, dinámica, normalización y pooling. No localiza cuál de esos componentes lo causa ni demuestra una imposibilidad biológica o de otras interfaces. Las escalas originales se conservaron: también forman parte de la representación comparada. Igual presupuesto adicional de los lectores no implica igual cómputo histórico; el cerebro ya estaba entrenado y quedó congelado. El holdout histórico se ha reutilizado; la evidencia nueva es la evaluación final.

Las cifras no son una comparación directa con el ~20% de los lectores refinados con DAgger y otros benchmarks. Las etiquetas del maestro se usan al entrenar y para registrar métricas; las decisiones finales no reciben un filtro de casillas seguras. La máscara de acciones solo excluye abiertas/fuera del tablero.

## Métricas por trayectoria

```json
{json.dumps(results, indent=2)}
```

Las oportunidades seguras y minas deducibles elegidas dependen de trayectorias distintas; comparar sus denominadores. No convertir las muertes de azar en victorias.

Auditoría: reconstrucción de500 layouts, solapamiento previo0, elección recalculada, conteos individuales cotejados, sello y hashes originales intactos. Metadatos del caché exactamente iguales a las filas y16 mapas recomputados contra el checkpoint. Se conservan pesos/Adam/RNG mejores y últimos. Ningún agente servido sustituido. Protocolo: `research/PROTOCOLO_COMPARACION_ENTRADAS.md`. Artefactos: `runs/input-representation-001`.
'''
    Path('research/RESULTADO_COMPARACION_ENTRADAS.md').write_text(doc)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
