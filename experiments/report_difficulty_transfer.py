"""Audit independent layout identities, count metrics, and graph each difficulty."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.evaluate_difficulty_transfer import OUT, SEEDS, LEVELS, N, prior_layouts
from experiments.train_risk_auxiliary import digest
from experiments.capacity_probe import write_json
from experiments.scaled_data import identity
from minesweeper import Minesweeper


def main(out=OUT):
    global OUT
    OUT = Path(out)
    assert json.loads((OUT/'completed.json').read_text())['completed']
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert all(digest(p) == sha for p, sha in manifest['hashes'].items())
    if manifest.get('expanded_mapping'):
        mapping_check = json.loads((OUT/'mapping-verification.json').read_text())
        assert mapping_check['old_5_7_exact']
        assert mapping_check['mapping_sha256'] == digest(OUT/'extended-mapping.pkl')
    games = json.loads((OUT/'games.json').read_text())
    keys = {g['layout_hash'] for g in games}
    assert len(games) == len(keys) == N * len(LEVELS)
    assert not keys & prior_layouts(exclude=OUT)
    for g in games:
        assert identity(Minesweeper(g['seed'], g['size'], g['mines'])) == g['layout_hash']
    metrics = ('safe_choices', 'safe_opportunities', 'known_mine_choices',
               'death_with_safe_available', 'death_after_exact_half_min_risk')
    summary, win_arrays = {}, {}
    for size, mines in LEVELS:
        expected = sorted((g['seed'], g['layout_hash']) for g in games if g['size'] == size)
        automatic, per_seed, wins = None, {}, []
        for seed in SEEDS:
            rows = json.loads((OUT/f'{seed}-{size}-games.json').read_text())
            assert [(r['seed'], r['layout_hash']) for r in rows] == expected
            assert all(r['size'] == size and r['mines'] == mines for r in rows)
            auto = [r['automatic'] for r in rows]
            if automatic is None:
                automatic = auto
            assert auto == automatic
            played = [r for r in rows if not r['automatic']]
            for r in played:
                assert 0 <= r['safe_choices'] <= r['safe_opportunities'] <= r['clicks']
                assert not (r['won'] and (r['death_with_safe_available'] or r['death_after_exact_half_min_risk']))
            w = np.array([int(r['won']) for r in played])
            wins.append(w)
            counts = {m: int(sum(r[m] for r in played)) for m in metrics}
            counts.update(wins=int(w.sum()), n=len(played), automatic_excluded=sum(auto))
            counts['missed_safe_choices'] = counts['safe_opportunities'] - counts['safe_choices']
            per_seed[str(seed)] = counts
        matrix = np.array(wins)
        win_arrays[size] = matrix
        mean_by_layout = matrix.mean(axis=0)
        rng = np.random.default_rng(20261027 + size)
        boot = [rng.choice(mean_by_layout, len(mean_by_layout), replace=True).mean()*100 for _ in range(10000)]
        summary[str(size)] = dict(mines=mines, per_seed=per_seed,
            mean_win_percent=float(mean_by_layout.mean()*100),
            conditional_ci95_percent=np.quantile(boot, [.025, .975]).tolist())
    write_json(OUT/'summary.json', summary)
    write_json(OUT/'independent_verification.json', dict(layouts_reconstructed=len(games),
        prior_overlap=0, originals_intact=True, identities_and_counts_checked=True,
        evaluation_valid_for_sizes=[s for s, _ in LEVELS] if manifest.get('expanded_mapping') else [7],
        legacy_zero_map_issue=not manifest.get('expanded_mapping')))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    x = np.arange(len(LEVELS))
    for i, seed in enumerate(SEEDS):
        counts = [summary[str(s)]['per_seed'][str(seed)] for s, _ in LEVELS]
        series = [
            [100*c['wins']/c['n'] for c in counts],
            [100*c['safe_choices']/max(1, c['safe_opportunities']) for c in counts],
            [100*c['death_with_safe_available']/max(1, c['n']-c['wins']) for c in counts]]
        for ax, y in zip(axes, series):
            ax.plot(x, y, 'o-', label=str(seed))
    for ax, title in zip(axes, ['Victorias autónomas', 'Elección segura cuando disponible', 'Derrotas con segura disponible']):
        ax.set_title(title)
        ax.set_xticks(x, [f'{s}×{s}\n{m} minas' for s, m in LEVELS])
        ax.set_ylim(0, 100)
        ax.set_ylabel('%')
        ax.grid(alpha=.2)
    axes[0].legend(title='Semilla de política')
    suffix = '-corrected' if manifest.get('expanded_mapping') else ''
    if not manifest.get('expanded_mapping'):
        fig.suptitle('INVÁLIDO para tamaños >7: entrada de actividad vacía')
    fig.savefig(f'research/difficulty-transfer{suffix}-results.png', dpi=150)
    plt.close(fig)
    table = []
    for size, _ in LEVELS:
        v = summary[str(size)]
        counts = ' / '.join(f"{v['per_seed'][str(s)]['wins']}/{v['per_seed'][str(s)]['n']}" for s in SEEDS)
        lo, hi = v['conditional_ci95_percent']
        table.append(f"| {size}×{size}/{v['mines']} | {counts} | {v['mean_win_percent']:.1f}% [{lo:.1f}, {hi:.1f}] |")
    validity = ('Mapeo ampliado explícitamente a 9/12/16, con igualdad exacta del mapeo histórico 5/7. '
        'No hubo entrenamiento con tableros grandes: mide transferencia con una interfaz ampliada. '
        if manifest.get('expanded_mapping') else
        '**INVALIDADA PARA TAMAÑOS MAYORES A 7.** La extracción histórica solo agrupaba actividad para 5/7; '
        '9/12/16 recibieron mapas cero. Sus resultados no miden generalización del cerebro. '
        'La auditoría de conteos no detectaba este error semántico. Se conservan datos como diagnóstico del fallo. ')
    Path(f'research/RESULTADO_TRANSFERENCIA_DIFICULTAD{suffix.upper()}.md').write_text(
        '# Evaluación por dificultad\n\n' + validity + '\n\n'
        '| Tablero/minas | Victorias por política (02 / 03 / 04) | Media e IC95% condicional |\n'
        '|---|---|---|\n' + '\n'.join(table) + '\n\n'
        f'![Resultados](difficulty-transfer{suffix}-results.png)\n\n'
        'Políticas fijadas antes del test, sin entrenamiento ni intervención del maestro. '
        '800 layouts nuevos, 200 por tamaño, compartidos por tres políticas sobre un cerebro. '
        'Victorias automáticas excluidas; el denominador se muestra por política. '
        'El intervalo remuestrea layouts conjuntamente entre políticas; no mide variabilidad entre cerebros. '
        'Con cero victorias el bootstrap es degenerado [0,0] y no demuestra probabilidad verdadera cero. '
        'Densidad cercana a 15%, tamaños y número de minas cambian juntos. '
        'El solver solo certifica un subconjunto de lo deducible. Las derrotas tras riesgo exacto de 50% '
        'se cuentan aparte y no se convierten en victorias.\n\n'
        'Auditoría de identidades, separación de layouts, conteos y hashes completada. '
        'No se cambiaron checkpoints ni agente servido.\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
