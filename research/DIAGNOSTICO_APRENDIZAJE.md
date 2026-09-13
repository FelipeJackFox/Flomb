# Diagnóstico del aprendizaje — 12 septiembre 2026

El híbrido queda pausado en14,384. El SHA256 del checkpoint original se comprobó sin cambios después de ambas pruebas. No se reanudó la corrida larga.

## Memorización y generalización

Pesos del híbrido, cabeza y cerebro completo; Adam nuevo solo en la copia supervisada.100posiciones de100tableros diferentes (50de5x5/3 y50de7x7/7), todas con al menos una jugada demostrablemente segura.100posiciones nuevas con semillas disjuntas para prueba. Las etiquetas provienen exclusivamente de información visible y total de minas. Se verifica que todas señalan acciones legales.

| Posiciones | Antes | Después |
|---|---:|---:|
| Conocidas |83/100|99/100|
| Nuevas |78/100|79/100|

250actualizaciones supervisadas en149.6segundos,165,181ganancias neuronales cambiaron. Esto demuestra capacidad de ajustar este conjunto y un camino de aprendizaje funcional; no descarta todos los errores posibles. La prueba nueva no muestra mejora clara. No es win rate: son decisiones individuales, principalmente posiciones tempranas con jugadas seguras. La pérdida distribuye su objetivo entre todas las casillas seguras.

## DAgger sin Q-learning

256episodios exactos5x5/3 y7x7/7, mezcla50/50,288actualizaciones. Inicialización cerebral común del piloto y cabeza nueva; no se parte del checkpoint memorizado ni es comparación causal emparejada con las14kpartidas del híbrido. Evaluación autónoma con32tableros fijos por estrato antes y después.

| Dificultad | Victorias antes | Victorias después | Seguras aprovechadas antes | Después |
|---|---:|---:|---:|---:|
|5x5-3|1/31|5/31|29/63|50/80|
|7x7-7|0/32|0/32|31/92|12/38|
|9x9-10|0/32|0/32|12/72|32/65|
|12x12-24|0/32|0/32|17/84|18/66|
|16x16-40|0/32|0/32|11/72|35/83|
|16x16-56|0/32|0/32|0/22|2/26|

Se excluyen aperturas automáticamente ganadas. Las oportunidades seguras dependen de las trayectorias visitadas: sus denominadores no son una prueba fija de posiciones. La evaluación intermedia128dio6/31en5x5; la final5/31, así que no hay progreso monotónico. La muestra y presupuesto no permiten declarar DAgger mejor que el híbrido ni demostrar que Q-learning causa el problema.

## Decisión

No ampliar dificultad ni lanzar otra corrida larga con esta evidencia. El siguiente experimento razonable es medir generalización de jugadas seguras sobre un conjunto mayor y diverso, y comparar una representación espacial del tablero con la entrada actual usando presupuestos e inicializaciones controlados. La hipótesis de representación sigue abierta; memorizar99/100 no la demuestra ni la descarta.

Archivos reproducibles: experiments/diagnose_learning.py; runs/learning-diagnostic-001/positions.npz, result.json y supervised-checkpoint.pkl; runs/diagnostic-dagger-001/config.json, fuentes de invocación, evaluaciones y checkpoint.pkl.18pruebas de núcleo e instrumentación pasaron.
