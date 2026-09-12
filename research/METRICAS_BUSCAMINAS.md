# Instrumentación específica de Buscaminas

Consulta de fuentes primarias: 12 de septiembre de 2026. Implementación local:
`experiments/game_metrics.py`; verificación: `python -m unittest experiments.test_game_metrics`.
Este módulo recibe únicamente números visibles y el número total de minas. No
recibe el entorno, las posiciones ocultas, el tablero resuelto ni una recompensa.
Las métricas describen la conducta; no se añaden al objetivo de entrenamiento.

## Qué medimos ahora

| Curva | Numerador / denominador | Qué puede concluirse |
|---|---|---|
| Aprovecha una casilla segura | `safe_taken / safe_opportunities` | Oportunidades encontradas por nuestro solver en que eligió alguna casilla demostrablemente segura; no coincidencia con una única acción arbitraria del maestro. |
| Clic en mina demostrada | `proven_mine_clicks / analyzed_moves` | Errores evitables que podían identificarse con la información pública. |
| Cobertura exacta | `exact_moves / analyzed_moves` | Fracción en que se pudieron enumerar las configuraciones consistentes; mostrar junto a todas las curvas de riesgo. No es porcentaje de exactitud de la política. |
| Riesgo inmediato elegido/mínimo | `chosen_risk_sum`, `min_risk_sum`, `excess_risk_sum`, cada uno sobre `exact_moves` | Diferencia de riesgo inmediato; **no** regret respecto a una política óptima de victoria. |
| Frontera / interior | respectivos clics sobre `analyzed_moves` | Si busca junto a pistas o en una zona sin vecinos abiertos. Interior significa fuera de la frontera, no lejos del borde geométrico. |
| Borde geométrico | `border_clicks / analyzed_moves` | Preferencia espacial; puede solaparse con frontera o interior. Por sí sola no indica aprendizaje. |
| Expansión por clic | `revealed_cells / reveal_observations`; `flood_clicks / reveal_observations` | Rendimiento de apertura y frecuencia de expansión automática. Excluye la apertura inicial automática; incluye cero en una explosión. |
| Supervivencia de apuestas exactas | supervivencias observadas frente a `exact_guess_expected_survivals / exact_guess_observations` | Si la suerte reciente ha sido peor/mejor que el riesgo visible. Registrar también la suma `p*(1-p)` y tamaño muestral. |
| Finales 50/50 certificados | `final_5050_certified`, resultado bruto y score separado | Reduce el ruido de una clase muy estrecha de finales; no borra derrotas ni certifica juego óptimo. |

Guardar todos los contadores en cada episodio y agregarlos por tamaño, número de
minas, densidad y fase del currículo. **Sumar numeradores y denominadores**, no
promediar los porcentajes de episodios de distinta longitud. Mostrar `null` cuando
no existe denominador; nunca dibujar eso como cero. Mantener curvas independientes
para evaluación autónoma, acciones del alumno durante entrenamiento y acciones
forzadas por el maestro. La ayuda no puede contar como competencia autónoma.

La API preacción es `analyze_move(visible, size, mine_count, action, analysis=None)`.
El argumento opcional reutiliza el `Analysis` del maestro para ese mismo estado
visible y número de minas, evitando repetir el solver. El llamador debe garantizar
que no proviene de otro estado; se verifican las casillas legales, no se repite
la enumeración para comprobar su procedencia. `visible` es
el tablero nativo sin padding, con -1 para ocultas y 0..8 para pistas. `action`
también está indexada sobre ese tamaño nativo. Tras ejecutar el clic, añadir
`revealed_delta` y `survived`. Al acabar, llamar a
`summarize_moves(moves, won, automatic_win=False)`. Si se muestrean movimientos,
añadir `{'metrics_sampled': True}` a la lista: invalida certificaciones de historia
completa. El muestreo debe registrar su tasa y ser independiente del resultado.

## Tratar el 50/50 sin fabricar victorias

Nuestro detector conservador acepta exclusivamente el último movimiento de una
partida completa con exactamente **dos casillas indeterminadas después de excluir
las minas demostradas**, una mina restante entre ellas, ambas con probabilidad
exacta 0.5 y ninguna acción segura pendiente. La acción elegida debe ser una de
esas dos. Exige además ausencia de infracción lógica detectada en la trayectoria
y resultado conocido del clic. Las minas demostradas siguen ocultas en nuestro
entorno, por eso se excluyen del conteo pero no se requiere colocación de banderas.
No es un detector general de apuestas inevitables en fases anteriores de la partida.

El resultado bruto sigue siendo 0 o 1. La segunda curva reemplaza **tanto victorias
como derrotas** de ese caso certificado por 0.5. Equivale a la esperanza del
resultado en ese final bajo el generador uniforme de tableros, y no a decir que
se ganó. Corregir solamente derrotas inflaría el resultado. No se eliminan partidas
con varias apuestas ni se divide el resultado por un producto de supervivencias:
eso introduce selección, varianza y supuestos adicionales. Una probabilidad
heurística de .5 nunca se convierte en final inevitable.

David Hill distingue 50/50 reales, pseudo-50/50 y casillas que no aportan nueva
información. Su motor además compara probabilidad de progreso y supervivencia de
la siguiente jugada: una acción algo más arriesgada puede mejorar las opciones
posteriores. Esto justifica no llamar «error estratégico» a toda diferencia con
el mínimo riesgo inmediato. [JSMinesweeper, documentación del autor](https://github.com/DavidNHill/JSMinesweeper#how-the-solver-determines-the-best-play).

## Qué investigar después y límites de interpretación

**Progreso lógico después de apostar.** Registrar qué proporción de apuestas
sobrevividas produce nuevas acciones demostrablemente seguras, descontando las
que ya estaban disponibles. Es una aproximación observable a obtener información,
no entropía ni causalidad. Implementar después un evaluador offline que enumere
las posibles pistas de cada acción permitiría estimar progreso esperado. Nuestro
módulo actual registra expansión, pero no calcula ese contrafactual ni ganancia de
información. Johnny Deuss explica explícitamente que probabilidad mínima de mina
y probabilidad máxima de ganar la partida son objetivos diferentes. Su solver
separa componentes y agrupa casillas con restricciones iguales, estrategia útil
para ampliar nuestra cobertura exacta con menor coste.
[Solver del autor](https://github.com/JohnnyDeuss/minesweeper-solver#how-it-works).

**Familias de deducción.** Conviene medir «cero restantes», «todas minas»,
«subconjuntos» y «conteo global» con denominadores de oportunidades. Nuestro solver
actual no devuelve una prueba ni la regla usada; por eso no etiquetamos sus
respuestas como «aprendió 1-2-1». Contar patrones dibujados sin verificar sus
restricciones podría atribuir razonamiento inexistente. La investigación de
Lordeiro y colaboradores estudia patrones locales y bandits, con especial éxito
en tableros pequeños; recomienda comprobar generalización por tamaño en vez de
inferirla de una curva global.
[Multi-Armed Bandits for Minesweeper](https://arxiv.org/abs/2007.12824).

**3BV y eficiencia.** 3BV caracteriza trabajo mínimo de apertura de un tablero
conociendo su estructura; sirve para estratificar dificultad de apertura, no
mide por sí solo dificultad lógica o riesgo. Requiere acceso a la estructura
completa y debe calcularse solamente en el evaluador aislado, nunca entregarse a
la política. No se implementó en este módulo visible-only. El ecosistema
MetaSweeper incluye 3BV/s y otras métricas de eficiencia, además de generadores y
análisis de partidas. Aquí los segundos del mouse son una animación y no tiempo
de razonamiento: usar clics y coste de inferencia, no 3BV/s de la escena.
[MetaSweeper, herramientas originales](https://github.com/Minesweeper-World/Solvable-Minesweeper).

**No-guess como suite separada.** Una suite cuya apertura y trayectoria de
resolución estén certificadas permite aislar fallos de razonamiento. Seleccionar
esos tableros cambia la distribución; no reemplaza el conjunto aleatorio normal
ni garantiza que cualquier secuencia de acciones siga siendo segura. Hill
advierte que cambiar el clic inicial puede invalidar la propiedad no-guess y
separa inicio seguro de inicio con apertura expandida.
[JSMinesweeper, opciones de partida](https://github.com/DavidNHill/JSMinesweeper#how-to-use-the-player).

**Límites computacionales.** Enumerar configuraciones compatibles puede ser
costoso. Nuestro `solver.analyze` conserva su cota de 12 casillas en la frontera
indeterminada; al superarla, no se certifican probabilidades. Los resultados de
complejidad de de Bondt refuerzan que no debemos vender un evaluador acotado como
oráculo universal. [The computational complexity of Minesweeper](https://arxiv.org/abs/1204.4659).

## Evaluación y saturación

Usar un conjunto fijo y reservado con las mismas semillas/aperturas en cada
checkpoint, sin exploración ni maestro. Separar 5x5, 7x7, 9x9, 12x12 y 16x16, y
en este último al menos 40 frente a 56 minas. Registrar victorias automáticas
aparte. Añadir intervalos binomiales al win rate y mostrar número de partidas;
para diferencias entre checkpoints usar evaluación pareada sobre las mismas
semillas. Una curva de entrenamiento puede caer porque sube la densidad, no
porque el modelo empeore. Una meseta breve de muestras pequeñas tampoco prueba
saturación.

Al cambiar los splits, conservar revisión, momento y mezcla aplicada, y marcarlo
en las curvas. La evaluación fija permanece sin cambios. Mantener muestras de
niveles anteriores para detectar olvido. Estas son recomendaciones experimentales
propias; no resultados ya demostrados para nuestro modelo.
