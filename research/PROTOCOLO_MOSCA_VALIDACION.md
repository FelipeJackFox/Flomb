# Validación de la mosca marcadora (prefijado 19 sept 2026, antes de evaluar)

Motivo (auditoría independiente + mis propias limitaciones declaradas): (a) en fly-marker-001 el control barajado consumía el mismo RNG
antes de sortear la firma olfativa, así que real-s y barajado-s diferían en cableado Y en olores; (b) los IC tomaban el tablero como unidad
cuando la afirmación es sobre semillas; (c) faltaba una línea base sin cerebro sobre los mismos sentidos; (d) la unión de victorias de 3 moscas
(66%) es una cota oráculo, no un ensamble jugable.

Cambios de código: permutaciones del barajado desde un generador propio (experiments/mushroom_body.py), interfaz de agentes
(experiments/fly_agents.py) verificada contra fly-marker-001 (real-0: 284/500 y pesos idénticos), pruebas experiments/test_fly.py (6 pasan).

Diseño: misma configuración que fly-marker-001 (rpe, sparsity .2, tuning .35, beta 100, eta .05/(1+g/3000), T 1.5, dread .03, 10,000 partidas).
- real-s y shuffled-s para s = 0..9: MISMA semilla de sentidos s (firma, sintonía, exploración) y solo cambia el cableado → contraste pareado por semilla.
- logistic-s, s = 0..9: sin cerebro; un peso libre por valor de olfateo + sesgo, regla delta con lr .3/(1+g/3000), mismo reflejo de marca y exploración.
- naive, hand-policy. Ensambles JUGADOS (media de valencias y de seguridad del peor olfateo): ens-real-3 (semillas 0-2), ens-real-10, ens-shuffled-10.
750 layouts reservados NUEVOS (disjuntos de 001). Unidad de análisis = semilla: media, desviación, IC95 t (9 gl); real − barajado con t pareada;
real − logística con t de Welch. Se reporta además la correlación entre valencia aprendida en peligro 0.125 y victorias a través de las 20 moscas.
Hipótesis a poner a prueba: H1 el cableado real no difiere del barajado; H2 la logística iguala a la mosca; H3 el cruce pista débil/casilla sin olor
explica la varianza entre semillas; H4 un ensamble jugado supera a la mosca individual y a la política a mano.
