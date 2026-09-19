# La mosca que olfatea y marca (prefijado 19 sept 2026, antes de evaluar en layouts reservados)

Enfoque de Felipe: cuerpo de adorno, cerebro recibiendo estímulos que una mosca conoce, aprendizaje solo por castigo/recompensa biológicos.
Sin backprop, sin encoder ni lector entrenados. Circuito: cuerpo pedunculado de MaleCNS con conteos crudos de sinapsis
(88 glomérulos/314 PN → 4,064 KC → 97 MBON; PAM 316, PPL1 16). Signo de cada MBON leído del cableado. Solo cambian KC→MBON.

Arena (decisiones de ingeniería, declaradas): una pista N reparte N unidades de olor "peligro" entre sus vecinas tapadas no marcadas;
cada casilla abierta huele a "abierto". Parada en una casilla tapada, la mosca olfatea por separado a cada vecina abierta (hasta 8 olfateos);
la valencia de la casilla es la suma de las respuestas MBON a sus olfateos. Una casilla sin vecinas abiertas no huele a nada: solo
aprende un término tónico de contexto. Mina = calor → PPL1; casilla segura = azúcar → PAM. Regla local de tres factores con dopamina
modulada por la predicción (realimentación MBON→DAN). Reflejo innato: si UN olfateo predice seguridad < 3%, la mosca deja una marca de
estrés en esa casilla; la marca absorbe una unidad del peligro de las pistas vecinas (semántica de bandera) y la mosca no la pisa.
La división N/k y la resta de marcas las hace la física de la arena, no el cerebro: el cerebro aprende qué olores predicen calor o azúcar
y, como consecuencia de lo aprendido, qué marcar.

Configuración fija (elegida en pilotos con semillas dev 20M/21M, nunca con layouts reservados): rule=rpe, sparsity=0.2, tuning=0.35,
beta=100, eta0=0.05 con decaimiento 1/(1+g/3000), temperatura de exploración 1.5, dread=0.03, 10,000 partidas de entrenamiento
(alternando 7×7/7 y 9×9/12, semillas 10,000,000+g). Endpoint fijo: la mosca tras 10,000 partidas, sin selección.

Brazos: cableado real ×3 semillas (0,1,2); cableado barajado ×3 semillas (mismos conteos por KC y por MBON, parejas al azar);
mosca ingenua (sin aprender); sin marcas (dread=0) con cableado real semilla 0; política hecha a mano (techo de estos sentidos).
Evaluación: 750 layouts nuevos reservados (500 de 9×9/12, 250 de 7×7/7), sin exploración ni aprendizaje, excluyendo aperturas que ganan solas.
Principal: % victorias 9×9 de la mosca real (media de 3 semillas) con IC95 bootstrap por layout. Secundarios: real − barajado (pareado),
7×7, ingenua, sin marcas, techo, marcas falsas. Referencia histórica: línea con backprop 49/500 (9.8%) en 9×9 con una inferencia.
Se verifica que ningún layout de entrenamiento coincide con los reservados.
