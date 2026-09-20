# El mundo de sentidos de la mosca: qué es sentido y qué es trampa

19 sept 2026 · análisis, sin código nuevo ni corridas · para decidir qué sigue.
Números citados: `runs/fly-marker-001`, `runs/fly-marker-002`, pilotos en `benchmarks/` (ver `ESTADO.md`).

## 0. Resumen en cinco líneas

1. Hoy la arena hace dos pasos de solver: divide N/k y resta las marcas. Las dos son trampa. Por eso una logística sin cerebro empata (47.5% vs 44.5%).
2. El cambio que vuelve legítimo el resultado es barato de enunciar: que cada olfateo entregue **crudos** N (número de la pista), k (cuánto tapado la rodea) y m (cuántas marcas mías la rodean), con física genérica, y que la relación entre ellos la aprenda el cerebro.
3. Casi todo lo "más de mosca" (propiocepción, ritmo, saciedad) es legítimo pero **no mueve victorias**; sirve para realismo o como sustituto honesto de hiperparámetros.
4. Las ideas más tentadoras y más de mosca —probar la casilla con las patas, sentir calor cerca de la mina— son las peores: filtran el estado oculto. Son fraude, no sentido.
5. De nuestro lado, lo que más promete contra el colapso (1 de cada 6 aprendices nunca marca) es un **entrenamiento no letal** y currículum por densidad. El ensamble sirve para reportar, no es "una mosca".

## 1. El criterio, vuelto operativo

Cinco pruebas que le paso a cada candidato. Las tres primeras miden trampa; la cuarta mide si el cerebro hace falta; la quinta, si es de animal.

| Prueba | Pregunta | Si falla |
|---|---|---|
| T1 Estado oculto | ¿Para calcular el estímulo hay que saber dónde están las minas? | Fraude |
| T2 Física que conoce las reglas | ¿La arena usa algo más que el estado visible de la casilla que emite + física genérica (emitir, sumar, decaer con distancia)? | Trampa |
| T3 Política de una línea | ¿Una política casi óptima sobre ese estímulo es un umbral sobre un solo canal? | Trampa: no queda nada que aprender |
| T4 Lector lineal | ¿Una logística sobre los canales crudos empata al cerebro? | No es trampa, pero el cerebro es adorno para ese sentido |
| T5 Animal | ¿Existe la modalidad en un animal real? | Es de solver, no de mosca |

Escala que uso en las tablas:
**0** sentido limpio · **1** sentido con física genérica · **2** gris · **3** trampa · **4** fraude.

### Lo de hoy, pasado por el criterio

| Pieza actual | Nivel | Por qué |
|---|---|---|
| Olor "abierto" por casilla | 0 | Estado visible de la propia casilla. |
| Olfatear cada vecina por separado | 0–1 | Muestreo secuencial; un animal lo hace. Falta el mecanismo (viento, antenas), pero no regala nada. |
| **Peligro = N/k** | **3** | Falla T2 (la arena sabe cuáles vecinas están tapadas *y divide*) y T3 (umbral en 1.0 = mina segura). N/k es literalmente la probabilidad de mina según una sola pista. |
| **La marca resta una unidad a la pista** | **3** | Falla T2. En el buscaminas real la interfaz NO resta banderas a los números; esa resta la hace el jugador en la cabeza. Es un paso de propagación de restricciones regalado. |
| Reflejo de marcar con umbral 3% | 2 | No es un sentido: es un pedazo de política escrito por nosotros. La inferencia ("esto es mina") sí la aprende la mosca; el umbral y la acción son nuestros. Además es biestable (piloto de umbrales). |
| Olor lejano de densidad (minas−marcas)/tapadas | 3 | Mi propia idea del punto 2. Le daba el *prior* ya dividido. Además rindió peor (43–51% vs 53.6%). Descartada por las dos razones. |
| Sintonía en banda de los glomérulos | — | No es información, es representación. Legítima como ingeniería, pero conviene saber que la pusimos para que una curva 1-D fuera fácil. |
| Azúcar en cada pisada segura | 2 | Ya es moldeo de recompensa: el juego real solo dice ganaste/perdiste. Es fiel (cada pisada segura es objetivamente buena), pero hay que declararlo. |

## 2. La propuesta central: olfateos crudos (N, k, m)

**Física, toda genérica.** Cada casilla emite según su propio estado visible y nada más:
- una pista emite su olor con intensidad N;
- una casilla tapada emite "tapado";
- una casilla marcada por la mosca emite la feromona de la marca.

Los olores se suman en el aire sobre cada casilla, igual en las 8 direcciones. Cuando la mosca antena a una vecina abierta Y huele tres cantidades: N de Y, k = cuánto "tapado" hay alrededor de Y, m = cuánta feromona mía hay alrededor de Y. Ninguna emisión depende de otra casilla ni de las reglas: pasa T1 y T2. No hay umbral de un solo canal que resuelva nada: pasa T3.

**Qué tiene que aprender.** Que "N igual a k" quema siempre, que "N igual a m" alimenta siempre, y todo lo graduado en medio. Es decir: la división y la resta pasan de la arena al cerebro.

**T4, y aquí dudo de verdad.** Primero pensé que una logística aditiva volvería a empatar, porque "N = k" es el máximo de log N − log k y eso es aditivo. Pero una aditiva no puede tener *a la vez* la diagonal (N = k) en certeza y lo de junto (3 de 4 → 75% mina) bien graduado: para empujar la diagonal a "seguro mina" tiene que empujar lo demás a "seguro a salvo". Predicción: la logística aditiva queda mal calibrada y marca mal o no marca; una tabla de pares (conjunciones exactas) lo clava; la mosca, cuyas Kenyon son conjunciones al azar de ~6 entradas, queda en medio. **Ese experimento es el que diría si el cerebro hace trabajo.** Puede salir que no.

**Ganancia esperada.** En victorias: cero o negativa al principio (35–50% en 9×9, contra 44.5% hoy), con 5–10× más partidas, porque hay que aprender una superficie 3-D en vez de una curva 1-D. La ganancia es de otra clase: el resultado deja de ser trampa y la comparación con backprop se vuelve pareja.

**Costo.** Medio: reescribir `fly_senses.sniffs` (emisiones + suma), tres odorantes por olfateo en vez de uno, controles (logística aditiva, tabla de pares, cableado barajado), pruebas unitarias. 1–2 días. Sin cómputo caro.

## 3. Tabla de estímulos candidatos

Ganancia = puntos porcentuales en 9×9 sobre la mosca individual media (44.5%), a ojo y con mi incertidumbre declarada. "0" no es desprecio: varios valen por legitimidad o robustez, no por victorias.

### 3a. Memoria

| Estímulo | Qué información da | Qué tendría que aprender | Nivel y por qué | Ganancia | Costo |
|---|---|---|---|---|---|
| Traza de elegibilidad (las Kenyon activas hace poco siguen "marcadas" unos pasos) | Ninguna nueva sobre el tablero; conecta un resultado con lo olido antes | A quién darle crédito cuando el premio llega tarde | 0. Existe en la mosca (condicionamiento de traza). | 0 por sí sola; **habilita** premio al ganar, marcar como acción aprendida y azúcar escasa | Bajo |
| Memoria de trabajo del olfateo anterior (+ qué paso di) | El par (pista A, pista B, cómo están colocadas) | Conjunciones entre dos pistas: el patrón 1-2, la regla del subconjunto | 0–1. Es recordar, no recibir la relación resuelta | +0 a +5. Un solver con esas reglas sube de ~60% a ~70–75%; la mosca capturaría una fracción, con muchísimas partidas | Alto: secuencias, propiocepción, espacio de estímulos enorme |
| Mapa de dónde se quemó | Posiciones de minas de partidas pasadas | Nada útil: cada tablero es nuevo | Si los tableros se repiten es **trampa por memorización** (aprende layouts, no buscaminas). Si no se repiten, inútil | 0 | — |
| "Esto ya lo olí y no cambió" (habituación) | Que un olfateo es redundante | A no re-muestrear | 0 | 0 (hoy no hay costo de tiempo) | Bajo |

### 3b. Olores de segundo orden

| Estímulo | Qué da | Qué aprender | Nivel | Ganancia | Costo |
|---|---|---|---|---|---|
| k crudo por pista (propuesta central) | Cuánto tapado rodea a la pista | La relación N–k | 1 | 0 en victorias; legitimidad | Medio |
| m crudo por pista (feromona propia sumada) | Cuántas marcas mías rodean a la pista | La relación N–m y que mis marcas cuentan | 1 | 0; quita la resta regalada | Medio (va con la anterior) |
| Olor de "casillas compartidas" entre dos pistas | Que los desconocidos de A son subconjunto de los de B | Casi nada | **3.** Es la regla del subconjunto servida. La relación entre vecindarios *es* la deducción | +5 a +10 | Medio — **no hacerlo** |
| Difusión a dos casillas (olor de pistas lejanas, atenuado) | Contexto 5×5 crudo | Conjunciones lejanas | 1 | ≈0. Ya lo probamos en versión simbólica: la información llega a las Kenyon (AUC 0.77–0.84) y la regla dopaminérgica no la exprime (0–8/200) | Bajo |
| Peligro residual iterado (la arena propaga hasta estabilizar) | Probabilidades ya resueltas | Leer | **3–4.** Es el solver completo en el aire | +15 a +25 | — **fraude metodológico** |

### 3c. Cuánto queda, densidad

| Estímulo | Qué da | Qué aprender | Nivel | Ganancia | Costo |
|---|---|---|---|---|---|
| Cantidad de tablero tapado (oscuridad global / olor "tapado" de fondo) | Un escalar crudo | Que con poco tablero las casillas sin olor son más peligrosas | 0–1 | +0 a +2 sola | Bajo |
| Contador de minas como cantidad cruda | Minas restantes (el juego real lo muestra) | Combinarlo con lo tapado para estimar riesgo de adivinar | 1. Ojo: el contador real resta banderas; esa resta es de la interfaz del juego, no nuestra | +2 a +4 junto con la anterior. Mi sonda: adivinar con densidad 54/300 vs constante 35–41/300 | Bajo |
| Densidad ya dividida | El *prior* servido | Un umbral | **3** (mi olor lejano) | — | — descartado |

Con densidad fija al 15% estos dos sentidos son casi redundantes entre sí; solo se vuelven informativos si variamos la densidad en el entrenamiento (sección 4).

### 3d. Cuerpo y estado interno

| Estímulo | Qué da | Qué aprender | Nivel | Ganancia | Costo |
|---|---|---|---|---|---|
| Propiocepción + costo de moverse | Dónde estoy respecto a la frontera; caminar cuesta | A preferir casillas cercanas | 0 | **0 o negativa**: el juego no tiene reloj; un costo de caminar solo empuja a elegir peor. Vale para el visor y como sustrato de la memoria entre pistas | Medio |
| Hambre / saciedad | Nada del tablero | Nada; modula cuánto vale el azúcar y cuánto arriesga | 0. Real: el hambre abre la expresión de memoria apetitiva | 0. Sustituto honesto de la temperatura de exploración y del decaimiento de la tasa de aprendizaje, que hoy son números nuestros | Bajo |
| Azúcar proporcional a lo que se abre (cascada = banquete) | Magnitud del premio | Que abrir zonas de ceros vale más | 1. Fiel al juego, pero es moldeo | +0 a +2 | Bajo |
| Novedad **interna** (una MBON que responde a lo no familiar y se habitúa; existe: α′3) | Qué tan poco conozco este olor | A probar lo raro de vez en cuando | 0 si la calcula el cerebro con su propia experiencia | +3 a +6 en la media **si** cura el colapso (ataca la causa: quedarse sin muestras). Hipótesis, no dato | Medio |
| Olor de incertidumbre **desde la arena** ("esta casilla es ambigua") | Cuáles casillas son deducibles | Nada | **3–4.** Saber qué es ambiguo exige resolver | — | — no |
| Tiempo, ritmo, reloj circadiano | Pasos transcurridos | Nada útil: redundante con "cuánto tapado queda" | 0 | 0 | Bajo |
| "Tiempo desde el último azúcar" | Racha de hambre | Urgencia | 0; se funde con hambre | 0 | Bajo |

### 3e. Feromonas y lo social

| Estímulo | Qué da | Qué aprender | Nivel | Ganancia | Costo |
|---|---|---|---|---|---|
| Feromona de mis marcas, en la casilla | Que ahí marqué | A no pisar (hoy es innato) | 0 | 0 | Ya existe |
| Feromona de mis marcas, sumada alrededor de la pista (= m) | Ver 3b | Ver 3b | 1 | — | — |
| Olor de estrés de moscas muertas en ese tablero (el dSO/CO₂ es real) | Dónde murieron otras | A evitarlo | Biológicamente impecable y **trampa igual**: solo informa si el tablero se repite, y entonces se memoriza el layout | — | — no |
| Marcas de moscas anteriores | Igual | Igual | Igual | — | — no |

### 3f. Más de mosca que de solver — y por eso peligrosas

| Idea | Por qué tienta | Veredicto |
|---|---|---|
| Probar la casilla con las patas antes de pisar (las moscas saborean con los tarsos) | Es lo más mosca que hay | **4, fraude.** Lo que probaría es la mina. Filtra estado oculto aunque sea con ruido. Cambia el juego. |
| Calor o humedad cerca de las minas | Las moscas tienen termo e higrorreceptores | **4.** Mismo problema: gradiente del estado oculto. |
| Viento y dirección de la pluma (anemotaxis), comparar antenas izquierda/derecha | Da de dónde viene cada olor | 0–1. No aporta información nueva: sustituye nuestro supuesto de "olfateos separados" por un mecanismo. Vale por honestidad, no por victorias. |
| Visión: tapado = oscuro, borde de la frontera | El lóbulo óptico ya calcula bordes | 0–1. Otra vía para k. Equivalente al olor "tapado". |
| Codificar N por concentración, no por identidad | Weber–Fechner: 7 y 8 se confunden | 0. Una limitación realista que casi no cuesta (los números altos son raros). |
| Consolidar entre partidas ("sueño", repetición) | Las moscas consolidan durmiendo | No es un sentido: es un algoritmo nuestro. Legítimo si se declara. |

## 4. Lo que podemos hacer nosotros

| Acción | Qué cambia | ¿Trampa? | Ganancia esperada | Costo |
|---|---|---|---|---|
| **Entrenamiento no letal** (pisar mina quema pero la partida sigue; la evaluación sigue siendo letal) | Muchas más muestras de los olores temidos por partida | No: es régimen de entrenamiento, como choques suaves antes de la prueba. La mina descubierta queda visible, que es información legítima | **+3 a +7 en la media**, casi todo por eliminar colapsos (5 de 30 aprendices en 002). Es mi mejor apuesta contra la bimodalidad | Bajo |
| Currículum por densidad (pocas minas → más) | Los casos extremos aparecen pronto y seguido | No. El moldeo gradual es práctica estándar con animales | +2 a +5 vía menos colapsos; techo igual. Con sentidos crudos probablemente sea necesario | Bajo |
| Variar la densidad durante el entrenamiento | El valor de la casilla sin olor deja de ser un número clavado al 15% | No | −3 a 0 en el 15% de siempre; robustez fuera de ahí (hoy: desconocida, probablemente mala). Obliga a tener sentidos de densidad (3c) | Bajo |
| Currículum con posiciones escogidas por un solver | Situaciones "didácticas" | **2–3.** El solver entra por la selección de ejemplos. Escoger por criterios genéricos (tamaño, densidad) sí; por deducibilidad no | — | — |
| Moldeo con conocimiento del solver (premio por pisar lo deducible, castigo por adivinar habiendo segura) | La respuesta entra por el canal de recompensa | **3.** Es un maestro disfrazado. La línea de backprop usó demostraciones del solver: otra categoría, y hay que decirlo cuando se comparen | — | — no |
| Castigar marcas falsas | La arena le diría qué marcas están mal | **4.** Eso es estado oculto. En el juego real una bandera falsa se descubre por contradicción o muriendo | — | — no |
| Castigo más fuerte que el premio | Aprende más rápido lo aversivo | No es trampa, pero **ya falló**: ×3 y ×6 hacen que marque todo el tablero (0%); ×2 rescata a unas y hunde a otras | 0 | — descartado |
| Premio grande al ganar (apareamiento) | Señal del objetivo real | No; necesita traza de elegibilidad | +0 a +1 | Bajo |
| Exploración dirigida por novedad interna | Ver 3d | No | Ver 3d | Medio |
| Exploración dirigida por el solver | — | 3 | — | no |
| **Ensamble de 10 moscas** | Promedia valencias | No es trampa, pero **no es una mosca**: es un enjambre que vota, y las moscas no votan. Úsese como herramienta de reporte | Medido: 53–55% vs 44.5% | Ya existe |
| Seleccionar moscas en tableros de desarrollo (descartar las que no marcan) | Cría/selección | No, si la selección no toca los tableros reservados. El colapso se detecta sin ellos (marcas por partida = 0) | Medido entre las que marcan: 49–52% | Trivial |

## 5. Dónde estoy siendo duro conmigo

- **N/k fue idea mía y es trampa.** La defendí como "física de la arena". Es una arena que conoce las reglas.
- **La resta de marcas también.** La justifiqué como "semántica de bandera"; en el juego real esa resta es del jugador.
- **El olor lejano de densidad, igual**, y encima rindió peor.
- **El 49.9% original estaba inflado** por tomar el tablero como unidad; con semillas es 44.5% [33.1, 56.0].
- **Mi argumento de "transformación monótona"** para el castigo asimétrico estaba mal; la auditoría tenía razón sobre el término tónico.
- **Las ganancias de este documento son estimaciones a ojo.** Las únicas medidas: ensamble (+9 a +11), selección de las que marcan (+5 a +8), adivinar con densidad (en una política a mano, no en la mosca).
- **Riesgo real de la propuesta central:** que con sentidos crudos la mosca no aprenda a marcar nunca y quede en ~3%, como la versión simbólica. Si pasa, la conclusión honesta sería que el salto de hoy era 100% de la arena.

## 6. Lo que recomiendo, en orden

1. **Olfateos crudos (N, k, m)** con los tres controles (logística aditiva, tabla de pares, barajado). Es el único cambio que ataca el criterio de Felipe de frente. Aceptar de antemano que el número puede bajar.
2. **Entrenamiento no letal + currículum por densidad**, porque con sentidos crudos la escasez de muestras va a empeorar, y ya es la causa del colapso.
3. **Sentidos crudos de densidad (tapado restante, contador) + densidad variable**, juntos o ninguno.
4. **Traza de elegibilidad → marcar como acción aprendida**, para quitar el umbral de 3% puesto a mano. Arriesgado; después de 1–3.
5. **Novedad interna** como exploración, solo si 2 no cura el colapso.

**No hacer:** patas que prueban minas, gradientes térmicos, olor de incertidumbre, olor de casillas compartidas, peligro iterado, moldeo con solver, castigo a marcas falsas, feromonas entre tableros repetidos.
**Cosmético, para cuando toque el cuerpo visible:** propiocepción, viento, hambre, ritmo.

La memoria entre pistas (3a) es donde el cerebro tendría trabajo de verdad no lineal, y también lo más caro y lo de menor rendimiento por esfuerzo. La dejaría para después de saber si 1 sale.
