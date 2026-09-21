# ¿Dónde le gana un cerebro a una tabla? (prefijado 20 sept 2026, antes de evaluar)

Antecedente: con pocas muestras (entrenamiento letal) la mosca le saca 22 pp a la tabla; con muchas (no letal) la tabla le saca 8 pp. Dos mediciones:
A. CURVA DE EFICIENCIA. Mosca vs tabla, entrenamiento letal y no letal, olfateos crudos, densidad variable, reflejo 3%, inicio viejo al entrenar. 8 semillas. Se evalúa (letal, inicio en cascada, 300 layouts 9×9/12 reservados nuevos,
   sin aprender durante la evaluación) tras 250, 500, 1k, 2k, 4k, 8k y 16k partidas. Se reportan también las pisadas de entrenamiento acumuladas. Pregunta: ¿dónde se cruzan las curvas?
B. RUIDO SENSORIAL. Cada concentración olida se multiplica por exp(σ·ε), ε normal, remuestreada en cada olfateo, en entrenamiento y en evaluación; σ ∈ {0, 0.15, 0.30}. No letal, 10k partidas, 8 semillas, mismos 300 layouts.
   Aprendices: mosca (recibe la concentración continua; sus glomérulos con sintonía en banda la codifican sin redondear) · tabla-entera (redondea al entero más cercano: usa a su favor que los valores verdaderos son enteros, cosa que la mosca no sabe)
   · tabla-fina (casillas de 0.5). Hipótesis: la tabla se degrada más que la mosca al subir σ. Puede salir que no, sobre todo contra la tabla-entera, porque redondear es un buen filtro de ruido.
Unidad = semilla, IC95 t, contrastes pareados por semilla.
