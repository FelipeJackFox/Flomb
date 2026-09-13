"""Report structural trials and interventions, separating necessity from usefulness."""
import argparse,json,pickle
from pathlib import Path
from experiments.report_scaled_training import interval


def scores(games):
    result=[]
    for size in (5,7):
        rows=[g for g in games if g['size']==size and not g['automatic']]
        wins=sum(g['won'] for g in rows)
        result.append(f'{wins}/{len(rows)} ({wins/len(rows):.1%}; IC95% {interval(wins,len(rows))})')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('runs/structural-learning-001'))
    parser.add_argument('--output',type=Path,default=Path('research/RESULTADO_INTERFAZ_Y_PLASTICIDAD.md'))
    args=parser.parse_args();training=args.root/'training'
    manifest=json.loads((training/'manifest.json').read_text())
    prepared=json.loads((args.root/'prepared.json').read_text())
    rewiring=json.loads((args.root/'rewiring.json').read_text())
    data=pickle.loads((args.root/'dataset.pkl').read_bytes())
    results=json.loads((training/'results.json').read_text()) if (training/'results.json').exists() else []
    lines=['# Interfaz por casilla y plasticidad de conexiones','',
           f"Semilla {manifest['seed']}; {manifest['updates']} actualizaciones, batch {manifest['batch']} y minibatches idénticos. "
           f"{len(data['train'])} posiciones de entrenamiento; {len(data['holdout'])} de validación; 500 partidas finales nuevas.",'',
           'Se reutilizan entrenamiento/validación de la fase ampliada. Las partidas finales no comparten layouts '
           'con esos conjuntos ni con los pilotos espacial y de capacidad anteriores. Varias posiciones de validación '
           'pertenecen a un mismo tablero; no son ensayos independientes. La comparación tiene una semilla y no adopta modelos automáticamente.','',
           '| Modelo | Aciertos reservados antes → después | 5×5 victorias sin apertura automática | 7×7 victorias | Tiempo de entrenamiento |',
           '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['variant']} | {r['before']['correct']} → {r['holdout']['correct']}/{r['holdout']['n']} | "
                     +' | '.join(scores(r['games']))+f" | {r['training_seconds']/60:.2f} min |")
    if not (training/'completed.json').exists():
        progress=json.loads((training/'progress.json').read_text()) if (training/'progress.json').exists() else {}
        lines += ['',f'En curso. Estado al generar el informe: {progress}.']
    lines += ['','## Uso de decisiones seguras y plasticidad','',
              '| Modelo | Elecciones seguras / oportunidades | Muertes con opción segura | Ganancias de conexiones modificadas | Ganancias neuronales modificadas |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        gs=r['games']
        lines.append(f"| {r['variant']} | {sum(g['safe_choices'] for g in gs)}/{sum(g['safe_opportunities'] for g in gs)} | "
                     f"{sum(g['death_with_safe_available'] for g in gs)} | {r.get('changed_edges','—')} | {r.get('changed_nodes','—')} |")
    lines += ['','## Intervenciones sin reentrenar','',
              '| Modelo | Intervención | Aciertos reservados | 5×5 victorias | 7×7 victorias |','|---|---|---:|---:|---:|']
    for r in results:
        for mode,result in r['interventions'].items():
            lines.append(f"| {r['variant']} | {mode} | {result['holdout']['correct']}/{result['holdout']['n']} | "+' | '.join(scores(result['games']))+' |')
    lines += ['','## Qué prueba y qué no prueba','',
              f"- Los {prepared['nodes']:,} nodos y {prepared['edges']:,} conexiones permanecen en las variantes del conectoma. "
              'Todas las conexiones originales tienen una ganancia independiente entrenable en retina_plastic. '
              'Que una ganancia exista no implica que reciba gradiente en cada minibatch: el informe cuenta las realmente modificadas.',
              f"- Entrada en {prepared['input_neurons']} neuronas L1/L2/L3 usando columnas ópticas anotadas. "
              'La lectura por casilla interpola poblaciones Mi/Tm/T1/C3 y usa el mismo decoder en todas las posiciones. '
              'La entrada y lectura usan poblaciones distintas. El decoder recibe estado neuronal y contexto público constante por tablero, no pistas directas.',
              '- El mapeo de hexágonos a una pantalla cuadrada, la imagen idéntica para ambos ojos y los tres pasos de tasas son decisiones de ingeniería, no fisiología validada.',
              '- retina_fixed mantiene los pesos de conexiones fijos, pero aprende ganancias por neurona e interfaces. '
              'retina_plastic añade 25.6 millones de multiplicadores independientes, positivos y acotados: conserva signos y topología.',
              f"- El grafo de control aceptó {rewiring['accepted']:,} intercambios. Cambió {rewiring['fraction_changed_slots']:.1%} "
              'de las asignaciones fuente-peso originales. Conserva grados de entrada/salida, multiconjunto de pesos entrantes, '
              'signos por conexión y autoconexiones existentes. No conserva grados ponderados de salida ni constituye una aleatorización completa.',
              '- Anular toda la actividad produce scores iguales por casilla por construcción. La caída bajo zero solo verifica '
              'dependencia funcional del canal neuronal; no prueba una ventaja del conectoma biológico. Shuffle permuta la actividad entre neuronas sin cambiar su distribución global.',
              '- La comparación entrenada con rewired_plastic, con iguales interfaces y presupuesto, es el control para empezar a evaluar '
              'el aporte del cableado. Requiere réplicas antes de atribuir una ventaja a la anatomía.',
              '- No se reanuda el híbrido anterior ni se inicia DAgger automáticamente. Los datos supervisados aún cubren decisiones seguras certificadas, no un objetivo explícito de apuestas.']
    args.output.write_text('\n'.join(lines)+'\n');print(args.output)


if __name__=='__main__':main()
