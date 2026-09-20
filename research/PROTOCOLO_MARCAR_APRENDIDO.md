# Punto 4: marcar como acción aprendida (prefijado 20 sept 2026, antes de evaluar)

Hoy marcar es un reflejo innato con umbral puesto a mano (seguridad predicha < 3%). Aquí la mosca empieza SIN marcar nunca y tiene que descubrir cuándo conviene.
Mecanismo (experiments/fly_agents.LearnedMarker): marca una casilla con probabilidad σ((θ − z)/τ), z = log-odds de seguridad que su cerebro predice para el olfateo más temido.
Marcar no da calor ni azúcar; cada decisión marcar/no marcar deja una traza de elegibilidad que decae 0.9 por pisada, y la dopamina de las pisadas siguientes (azúcar +1, calor −3,
relativa a un promedio móvil) ajusta el único parámetro θ. Nadie le dice si una marca fue correcta. Declarado: +1/−3 es moldeo nuestro; τ=1, decaimiento 0.9 y tasa 0.005 se eligieron en piloto dev.
Base: olfateos crudos, entrenamiento no letal, densidad variable, 20,000 partidas, 10 semillas, unidad = semilla. Evaluación letal y determinista (marca si z < θ) en 750 layouts reservados nuevos (500 de 9×9/12, 250 de 7×7/7).
Brazos: fly-learned (θ0 = −8, no marca al inicio) · fly-reflex (umbral fijo 3%, lo de siempre) · fly-nomarks · fly-sham (igual que learned pero el calor/azúcar que recibe la compuerta es aleatorio con la misma
frecuencia: si θ sube igual, la subida es un artefacto y no crédito real) · table-learned y table-reflex (sin cerebro).
Principal: fly-learned − fly-nomarks en 9×9 (¿aprende a marcar?). Secundarios: fly-learned − fly-reflex (¿el umbral aprendido iguala o mejora al puesto a mano?), fly-learned − fly-sham, θ final y su dispersión,
marcas por partida, marcas falsas, aprendices que nunca marcan, tabla.
Piloto dev (3 semillas × 3 variantes, 15k partidas): θ sube de −8 a ≈ −6.3 (umbral ≈ 0.2%), 6–7 marcas por partida, 0 marcas falsas, 45–50% en 9×9 dev.
