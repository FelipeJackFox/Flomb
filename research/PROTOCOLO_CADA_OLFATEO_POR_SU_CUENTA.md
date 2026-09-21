# Cada olfateo aprende por su cuenta; la mosca promedia sus opiniones (prefijado 21 sept 2026, antes de evaluar)

Diagnóstico de la mosca hundida de runs/fly-aggregate-001 (semilla 5): al promediar, el error de aprendizaje es el del PROMEDIO de la casilla, así que un olfateo de mina segura tiene que volverse extremo para pesar
(log-odds de seguridad −31 para N=1,k=1). Ese extremo se contagia, por el solape de las Kenyon, a olfateos parecidos que casi siempre son seguros: (N=2,k=5,m=1) quedó en −7 y disparó 627 marcas falsas de 862. Aparece tarde: a 12k partidas esa semilla estaba sana.
Arreglo: cada olfateo de la casilla pisada se condiciona POR SEPARADO con el resultado (su propio error de predicción), y la decisión es la media de las opiniones. Ningún olfateo necesita exagerar. No añade información.
Piloto dev (benchmarks/fly-aggregate-pilot5, 20k partidas, semillas 0–9): cada-olfateo 79.5% en 9×9 (mín. 68.5, 42 marcas falsas) · media normal 73.3% (una semilla en 12%) · suma 73.2%. Otras variantes descartadas: suma/√n 74.2%; media + alarma 67.9% (se hunde una semilla).
FORMAL: semillas 10–19 (no usadas al diseñar el arreglo), 20k partidas, olfateos crudos, no letal, densidad variable, compuerta aprendida con tope θ ≤ 0; evaluación letal con cascada en layouts reservados nuevos (500 de 9×9, 250 de 7×7, 150 de 16×16).
Brazos: sum-cap · mean-cap · meanown-cap. Principal: meanown-cap − sum-cap en 9×9, pareado por semilla. Secundarios: 16×16, 7×7, peor semilla de cada brazo, marcas falsas, política a mano en los mismos layouts.
