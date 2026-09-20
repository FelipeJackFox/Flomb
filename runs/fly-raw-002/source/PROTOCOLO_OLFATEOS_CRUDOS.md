# Olfateos crudos (N, k, m) — evaluación formal (prefijado 19 sept 2026, antes de evaluar)

Pregunta de Felipe: si la arena deja de dividir N/k y de restar marcas, ¿la mosca aprende a marcar, o el salto era 100% de la arena?
Sentidos: experiments/fly_senses.sniffs_raw — cada casilla emite según su propio estado visible (pista: N; tapada: "tapado"; marcada: feromona),
los olores se suman igual en las 8 direcciones, y al antenar una vecina abierta la mosca huele N, k y m crudos. Marcar no altera ningún número.
Todo lo demás idéntico a fly-marker-002: rpe, sparsity .2, tuning .35, beta 100, eta .05/(1+g/3000), T 1.5, reflejo de marca al 3% por peor olfateo,
10,000 partidas (el piloto dev muestra meseta desde 5,000 y ninguna tendencia hasta 40,000), mismas semillas de entrenamiento 10,000,000+g.

Brazos, 10 semillas cada uno (0..9), unidad de análisis = semilla:
- raw-real y raw-shuffled (pareadas: misma semilla de sentidos, solo cambia el cableado);
- table: sin cerebro, un peso por triple (N,k,m) exacto; additive: sin cerebro ni conjunciones, a[N]+b[k]+c[m];
- share-real: la mosca con sentidos de trampa (N/k y resta) sobre ESTOS MISMOS tableros, como referencia de cuánto regalaba la arena;
- naive-raw (sin aprender), hand-policy (techo; la información cruda es equivalente), ens-raw-10 (enjambre que vota, no una mosca).
750 layouts reservados nuevos. Principal: % victorias 9×9 de raw-real, media e IC95 t por semilla. Secundarios: raw-real − share-real (cuánto era de la arena),
raw-real − table y − additive (¿el cerebro aporta?), raw-real − raw-shuffled pareado, fracción de aprendices que nunca marcan, 7×7, marcas falsas.
Criterio de lectura fijado de antemano: si raw-real queda en ≤5% en 9×9 (como la versión simbólica y la mosca sin marcas), se reporta "el salto era de la arena" sin matices.
Piloto dev previo (benchmarks/fly-raw-pilot, 150+100 tableros dev): mosca 20–40% en 9×9 con media ~30%, tabla 17–25% con parpadeo de marcas, aditiva 5–20% inestable.
