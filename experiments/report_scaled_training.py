"""Descriptive report with episode-level uncertainty, never treating states as independent boards."""
import argparse
import json
import math
from pathlib import Path


def interval(wins,n):
    if not n:
        return '—'
    z=1.96
    p=wins/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return f'{center-half:.1%}–{center+half:.1%}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('runs/scaled-learning-001'))
    parser.add_argument('--output',type=Path,default=Path('research/RESULTADO_ESCALA_Y_ABLACIONES.md'))
    args=parser.parse_args()
    summary=json.loads((args.root/'dataset-summary.json').read_text())
    lines=['# Escala de demostraciones y comparación de componentes','',
           f"{summary['train']['n']:,} posiciones de entrenamiento de {summary['train']['unique_layouts']} tableros; "
           f"{summary['holdout']['n']:,} posiciones reservadas de {summary['holdout']['unique_layouts']} tableros; "
           f"{summary['games']['n']} partidas finales independientes.", '',
           'Las 500 partidas finales no aparecen en los conjuntos de los dos pilotos de arquitectura anteriores. '
           'Los splits nuevos no comparten layouts y los estados de entrenamiento/validación son distintos. '
           'Varias posiciones pertenecen al mismo tablero: los aciertos por posición son descriptivos, no 1,000 ensayos independientes.', '']
    for subdir,title in (('cnn','Control CNN con mayor presupuesto'),('ablations','Comparación emparejada de componentes')):
        root=args.root/subdir
        if not (root/'manifest.json').exists():
            continue
        manifest=json.loads((root/'manifest.json').read_text())
        lines += [f'## {title}','',f"{manifest['updates']} actualizaciones, batch {manifest['batch']}, semillas {manifest['seeds']}.", '',
                  '| Modelo / semilla | Posiciones antes → después | 5×5 victorias* | IC 95% | 7×7 victorias* | IC 95% | Entrenamiento |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        results=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else []
        for r in results:
            counts=[]
            for size in (5,7):
                games=[g for g in r['games'] if g['size']==size and not g['automatic']]
                wins=sum(g['won'] for g in games)
                counts.extend([f'{wins}/{len(games)} ({wins/len(games):.1%})',interval(wins,len(games))])
            lines.append(f"| {r['variant']} / {r['seed']} | {r['before']['correct']} → {r['holdout']['correct']}/{r['holdout']['n']} | "
                         +' | '.join(counts)+f" | {r['training_seconds']:.1f} s |")
        if not (root/'completed.json').exists():
            progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
            lines += ['',f"En curso; último estado guardado al generar este informe: {progress}."]
        lines += ['', '*Se excluyen aperturas que ganaron automáticamente; IC de Wilson sobre partidas. '
                  'Las semillas reutilizan las mismas partidas y no se suman como tableros nuevos.', '',
                  '| Modelo / semilla | Jugadas seguras elegidas / oportunidades | Muertes con jugada segura disponible | Muertes tras elegir riesgo mínimo exacto 50% |',
                  '|---|---:|---:|---:|']
        for r in results:
            games=r['games']
            safe=sum(g['safe_choices'] for g in games)
            opportunities=sum(g['safe_opportunities'] for g in games)
            avoidable=sum(g['death_with_safe_available'] for g in games)
            half=sum(g['death_after_exact_half_min_risk'] for g in games)
            lines.append(f"| {r['variant']} / {r['seed']} | {safe}/{opportunities} | {avoidable} | {half} |")
        lines += ['']
    lines += ['## Interpretación y límites','',
              '- El control CNN largo usa más ejemplos procesados por actualización; no se compara su costo o precisión como si tuviera el mismo presupuesto que las ablaciones.',
              '- Dentro de `ablations`, CNN y cuatro variantes del conectoma reciben las mismas secuencias de minibatches. Las cuatro variantes del conectoma tienen un canal y la misma cabeza inicial: actual, solo encoder, solo dinámica interna, ambos. Mantienen todas las conexiones.',
              '- El cambio interno incluye sesgo, retención, reinyección y transformación compartida. Aísla ese conjunto del encoder, no cada mecanismo individual. La ganancia original continúa aprendiendo en las cuatro variantes.',
              '- Los ejemplos de entrenamiento tienen al menos una acción segura certificada. Las apuestas siguen sin objetivos explícitos de riesgo en esta fase.',
              '- El contador 50% exige probabilidades exactas del solver y que la elegida tenga riesgo inmediato mínimo de 50%. No prueba optimalidad futura ni convierte automáticamente una derrota en victoria.',
              '- Las oportunidades seguras las certifica un solver acotado: puede omitir deducciones. No se interpreta ausencia de certificado como imposibilidad matemática de una acción segura.',
              '- No se ha aislado todavía el efecto de aumentar datos frente a aumentar actualizaciones, ni el aporte de la topología biológica frente a un grafo de control.',
              '- Híbrido original pausado; no se migran estos checkpoints a su arquitectura.']
    args.output.write_text('\n'.join(lines)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
