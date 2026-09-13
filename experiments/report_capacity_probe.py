"""Summarize a completed capacity probe without choosing a winner from tiny strata."""
import argparse
import json
from pathlib import Path
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=Path('research/RESULTADO_CAPACIDAD_ENTRENABLE.md'))
    args = parser.parse_args()
    completed = json.loads((args.run/'completed.json').read_text())
    manifest = json.loads((args.run/'manifest.json').read_text())
    results = json.loads((args.run/'results.json').read_text())
    lines = ['# Comparación de capacidad entrenable', '',
             'Prueba exploratoria de arquitectura, no selección final de algoritmo RL.', '',
             f"{manifest['updates']} actualizaciones por modelo y semilla; minibatches idénticos de 16; "
             '512 posiciones de entrenamiento, 126 reservadas y 48 tableros de partidas autónomas. '
             'Los tres conjuntos no comparten distribuciones de minas; entrenamiento y posiciones reservadas tampoco comparten estados visibles.', '',
             '| Modelo | Semilla | Reservadas antes → después | Entrenamiento | Victorias 5×5* | Victorias 7×7* | Segundos de entrenamiento |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        games = {size:[g for g in r['games'] if g['size']==size and not g['automatic']] for size in (5,7)}
        wins = {size:f"{sum(g['won'] for g in gs)}/{len(gs)}" for size,gs in games.items()}
        lines.append(f"| {r['variant']} | {r['seed']} | {r['before']['correct']}/{r['before']['n']} → "
                     f"{r['holdout']['correct']}/{r['holdout']['n']} | {r['train']['correct']}/{r['train']['n']} | "
                     f"{wins[5]} | {wins[7]} | {r['training_seconds']:.2f} |")
    refs = args.run/'references.json'
    if refs.exists():
        for name, rows in json.loads(refs.read_text()).items():
            scores = []
            for size in (5, 7):
                gs = [g for g in rows if g['size']==size and not g['automatic']]
                scores.append(f"{sum(g['won'] for g in gs)}/{len(gs)}")
            lines.append(f"| {name} | referencia | — | — | {scores[0]} | {scores[1]} | — |")
    lines += ['', '*Se excluyen las partidas ganadas automáticamente por la apertura. '
              'Tiempo de entrenamiento excluye generación de datos, carga del grafo y evaluaciones.', '',
              f"Una acción legal uniforme tendría un acierto esperado del {results[0]['holdout']['random_safe_probability']:.1%} "
              'en las posiciones reservadas. El maestro es una referencia práctica, no un límite óptimo demostrado.', '',
              '## Decisiones reservadas por dificultad y tipo de deducción', '',
              '| Modelo / semilla | 5×5 elemental | 5×5 relacional/exacta | 7×7 elemental | 7×7 relacional/exacta |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        s = r['holdout']['strata']
        cells = [f"{s[k]['correct']}/{s[k]['n']}" for k in ('5/elementary','5/relational_or_exact','7/elementary','7/relational_or_exact')]
        lines.append(f"| {r['variant']} / {r['seed']} | " + ' | '.join(cells)+' |')
    lines += ['', 'Elemental significa que el estado admite al menos una deducción segura por cierre cero/completo, '
              'incluido el total público de minas. Relacional/exacta significa que el maestro certifica seguridad '
              'más allá de ese cierre. No es una lectura del razonamiento interno del modelo.', '',
              '## Comparaciones emparejadas', '']
    for seed in manifest['seeds']:
        by_name = {r['variant']:r for r in results if r['seed']==seed}
        if 'current' not in by_name:
            continue
        baseline = by_name['current']['holdout']['hits']
        for name in ('expressive','cnn'):
            if name not in by_name:
                continue
            candidate = by_name[name]['holdout']['hits']
            only_new = sum(a and not b for a,b in zip(candidate,baseline))
            only_old = sum(b and not a for a,b in zip(candidate,baseline))
            p = binomtest(only_new, only_new+only_old).pvalue if only_new+only_old else 1.
            lines.append(f'- {seed}, {name} frente a current: solo candidato acierta {only_new}; '
                         f'solo actual acierta {only_old}; McNemar exacto bilateral p={p:.4g}.')
    lines += ['', 'Valores p exploratorios sin ajuste por comparaciones múltiples. Las semillas repiten los mismos '
              'tableros: no deben sumarse como nuevas observaciones independientes.', '',
              '## Alcance y límites', '',
              '- `current` conserva la propagación original y su clase de política lineal. Se validó equivalencia '
              'de features y gradientes; usa una cabeza categórica y Adam comunes a la prueba, no la optimización QR-DQN original.',
              '- `expressive` conserva todo el grafo; aprende un codificador convolucional, dos canales por neurona, '
              'ganancias, sesgos, mezcla de canales y retención. Cambia varios componentes juntos. No aísla cuál ayuda.',
              '- `cnn` es un control convencional pequeño. No está igualado en parámetros ni costo con el conectoma. '
              'Todos reciben los mismos datos, etiquetas, actualizaciones y tasa de aprendizaje; no es una búsqueda de hiperparámetros.',
              '- El entrenamiento balancea tamaño y categoría. La evaluación conserva su distribución; '
              'solo hay dos posiciones relacionales/exactas 7×7. Ese estrato requiere ampliación antes de concluir.',
              '- Los ejemplos supervisados son estados con al menos una jugada segura certificada. '
              'Las partidas autónomas sí pueden necesitar apuestas; esta prueba no entrena ni valida probabilidades de riesgo.',
              '- El pico de memoria guardado es acumulado del proceso macOS, no consumo aislado de cada modelo.',
              '- Los checkpoints incluyen parámetros y optimizador. La restauración del optimizador pasó una prueba de igualdad exacta; '
              'estos archivos no son compatibles directamente con el entrenador híbrido.',
              '- Falta control con conectividad reconfigurada antes de atribuir una ventaja a la anatomía biológica.', '',
              f"Checkpoint híbrido preservado: {completed['protected_checkpoint_intact']}. Corrida principal sin reanudar.", '',
              f"Datos, resultados, manifiesto, fuentes exactas y checkpoints: `{args.run}`."]
    args.output.write_text('\n'.join(lines)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
