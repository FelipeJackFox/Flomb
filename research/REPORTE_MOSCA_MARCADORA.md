# Reporte: la mosca que olfatea y marca (19 sept 2026)

Fuentes: runs/fly-marker-001/summary.json, research/PROTOCOLO_MOSCA_MARCADORA.md, benchmarks/fly-marker/*.log (pilotos), ESTADO.md.

## 1. Resultado
Mosca simulada entrenada solo con calor (mina) y azúcar (casilla segura): 49.9% de victorias en 9×9/12 minas y 61.2% en 7×7/7 minas,
en 750 tableros reservados. Mosca ingenua: 0.0% / 0.4%. Línea previa con backprop: 9.8% en 9×9 (otros 500 tableros, una semilla, entradas distintas).

## 2. Qué aprende y cómo
- Circuito: cuerpo pedunculado de MaleCNS con conteos crudos de sinapsis: 314 PN (88 glomérulos) → 4,064 Kenyon → 97 MBON; 316 PAM, 16 PPL1. Modelo de tasas, sin spikes.
- Signo de cada MBON leído del cableado (más PPL1 = acercarse; más PAM = evitar). Coincide con la biología conocida (MBON01-10 evitar, MBON11-14 acercarse).
- Arena (ingeniería declarada): pista N reparte N unidades de olor "peligro" entre sus vecinas tapadas no marcadas (N/k); casillas abiertas huelen a "abierto".
  La mosca olfatea por separado a cada vecina abierta; valencia de la casilla = suma de respuestas MBON por olfateo. Casilla sin olor: solo un término tónico de contexto.
- Aprendizaje: mina → calor → PPL1; segura → azúcar → PAM, en cada pisada. Solo cambian Kenyon→MBON (27,558 de 61,210 en real-0), regla local de tres factores
  con dopamina modulada por sorpresa. Sin backprop, encoder ni lector.
- Marca: reflejo innato (no aprendido). Si UN olfateo predice seguridad < 3%, deja marca de estrés, no la pisa, y la marca absorbe una unidad del peligro de las
  pistas vecinas (semántica de bandera). Así aparecen casillas de peligro 0 = seguras con certeza.
- Curva aprendida (valencia ×100 por olfateo, real-0): peligro 0 → +7.4; 0.125 → +0.5; 0.25 → −0.7; 0.5 → −1.9; 1.0 → −6.2.

## 3. Setup
10,000 partidas de entrenamiento por mosca alternando 7×7/7 y 9×9/12 (semillas 10,000,000+g), exploración softmax (T=1.5), eta 0.05/(1+g/3000).
Evaluación sin explorar ni aprender en 500 tableros 9×9 + 250 de 7×7 reservados; 0 solapes con entrenamiento. Ganar = abrir todas las seguras; primera apertura
central automática igual para todos; ninguna partida se ganó sola. Brazos: real ×3 semillas, barajado ×3, ingenua, sin marcas, política a mano (techo).
Configuración elegida con pilotos en semillas dev, nunca con los reservados. Protocolo escrito antes de evaluar.

## 4. Números
| Brazo | 9×9 | 7×7 |
|---|---|---|
| Ingenua | 0.0% | 0.4% |
| Aprende, sin marcas | 2.4% | 12.4% |
| Real (media 3 semillas) | 49.9% [46.3, 53.5] | 61.2% [56.0, 66.4] |
| Barajado (media 3 semillas) | 55.9% [51.9, 60.0] | 63.3% [57.9, 68.7] |
| Techo a mano | 63.6% [59.4, 67.8] | 71.6% [66.0, 77.2] |

Por semilla 9×9: real 284 / 275 / 190 de 500; barajado 249 / 296 / 294. Real − barajado: −6.0 pp [−7.6, −4.4]. Marcas falsas: 0 en todos los brazos.
Pisadas seguras 95–98% (ingenua 67%).

## 5. Mejor y peor de lo esperado
Mejor: aprende en ~2,500 partidas, cero marcas falsas, ~78% del techo de sus sentidos. Peor: sentidos simbólicos (vecindario 5×5 como olores) no aprendieron a jugar
(0–8/200) aunque la información llegaba a las Kenyon (AUC supervisado 0.77–0.84; regla dopaminérgica 0.70–0.74); el cableado real no supera al barajado;
ventana de configuración estrecha (β=300 marca todo y gana 0; β=30 no marca; umbral 1% hace parpadear las marcas).

## 6. Limitaciones
1. Los IC son por tablero, no por semilla. Entre semillas: 56.8 / 55.0 / 38.0 → intervalo t con 3 semillas ≈ 50 ± 26.
2. "Real peor que barajado" no está establecido: la misma semilla controla barajado, firma olfativa y sintonía; real-s y barajado-s no comparten olores.
3. Contra backprop la comparación no es pareja: la mosca recibe N/k y la resta de marcas ya hechas por la arena. El cerebro aprende una función de una variable + umbral.
4. Falta la línea base obvia: regresión logística sobre estos mismos sentidos (casi seguro rinde igual).
5. Sin pruebas unitarias del código de la mosca; la corrección de la física se apoya en "cero marcas falsas" y en que la política a mano funciona.
6. Biología delgada: sin spikes; dopamina como fracciones por MBON; retroalimentación MBON→DAN como escalar; firma de olores, sintonía en banda y "marca de estrés" son supuestos.
Qué se rompería: densidad de minas distinta (el valor de la casilla sin olor es un escalar aprendido a ~15% de minas; sin noción de densidad global);
azúcar escasa o solo al ganar (no hay traza temporal: crédito solo a la casilla recién pisada; tampoco se modela cantidad de azúcar); sentidos ruidosos (umbral 3% supone olores exactos).
Tamaño de tablero: sentidos locales, debería aguantar; no probado.

## 7. ¿Una run más larga mejoraría? NO.
Evidencia:
- Curva plana desde las 2,500 partidas. Piloto con la config final, 200 tableros dev 9×9: 120, 117, 123, 122, 126, 123 victorias a las 2.5k, 5k, 7.5k, 10k, 12.5k y 15k partidas.
  El ruido de muestreo en 200 tableros es ±7, así que eso es una línea horizontal.
- La tasa de aprendizaje ya decayó 4.3× a las 10k partidas (0.05 → 0.0115). Más partidas casi no mueven las sinapsis.
- Las semillas buenas están en 55–59% con techo de estos sentidos en 63.6%: aunque todo saliera perfecto quedan 5–8 pp, y ese resto es adivinanza, no falta de entrenamiento.
- La semilla mala (real-2, 38%) NO está subentrenada ni falla al marcar: marca 6.1 casillas por partida (las buenas 7.6–8.0) y predice 0.5% de seguridad para peligro saturado.
  Su problema es de calibración: valora peligro bajo (0.125) en −1.2 cuando las buenas lo tienen en ~0, así que prefiere casillas sin olor (que valen 0) y adivina más.
  CORRECCIÓN a mi primer reporte: ahí sospeché "parpadeo de marcas"; los datos dicen que no.
Conclusión: el techo está en la representación y en los sentidos, no en el tiempo. Ganancia esperada de una run 10× más larga: 0 ± 2 pp en semillas buenas; incierta en la mala.

Qué sí movería la aguja (en orden):
1. Validez antes que rendimiento (~1 h en esta Mac; cada mosca ≈ 5–6 min, 9 en paralelo): regresión logística sobre los mismos sentidos + 10–20 semillas separando semilla de olores
   de semilla de cableado. No sube el número; dice cuánto vale.
2. Robustez entre semillas: calibrar peligro bajo contra la casilla sin olor (sintonía más fina en shares bajos, o un olor global de "cuánto tablero queda tapado" para que el
   contexto no sea un escalar fijo). Espero llevar las semillas malas de ~38% a ~55% → media +4 a +6 pp, y +2 a +4 pp extra por adivinar con densidad (la política a mano la usa).
3. Subir el techo: deducciones de dos pistas (patrones 1-2) requieren un olor de segundo orden; un solver completo anda por 80%+. Espero +10 a +15 pp,
   al costo de meter todavía más razonamiento en la arena y menos en el cerebro.
4. Que el umbral de marca se aprenda en vez de ser innato: no sube el número, pero quita la pieza más "puesta a mano".

## 8. Addendum 19 sept (validación 002 y punto 2) — corrige números de arriba
- Con 10 semillas y la semilla como unidad (750 layouts reservados nuevos): real 44.5% IC95 [33.1, 56.0] en 9×9 y 60.8% [48.5, 73.1] en 7×7. El 49.9% [46.3, 53.5] de la sección 4 era optimista.
- Barajado pareado 35.3% [19.0, 51.6]; real − barajado +9.3 pp [−6.2, +24.7], p=0.21: no se detecta diferencia. Entre moscas que marcan: 49.2 vs 49.1.
- Logística sin cerebro, mismos sentidos: 47.5% [36.2, 58.9]; entre las que marcan 52.5% (DE 2.3) vs 49.2% (DE 6.4) de la mosca. El circuito no aporta nada medible sobre un aprendiz lineal.
- Resultado bimodal: 5 de 30 aprendices nunca marcan y se quedan en 2–4% (su predicción para peligro 1.0 quedó en 3.4–8.9%, justo sobre el umbral de 3%).
- Ensambles jugados: 3 moscas 46.8%, 10 moscas 53.2%, 10 barajadas 55.4%; política a mano 57.8% en estos layouts. El "66%" era la unión oráculo, no un voto.
- Punto 2: ninguna variante (tónico como referencia, olor lejano de densidad, umbral 10/20%, castigo asimétrico ×2/×3/×6) mejora a la configuración actual en pilotos dev; varias crean marcas falsas o hacen que la mosca marque todo el tablero. El reflejo de marca con umbral fijo es biestable.
- Corrección a la sección 7: para las moscas sanas sigo diciendo NO a la run larga; para las colapsadas podría destrabar el marcado, pero lento, porque dejaron de recibir muestras del olor temido.
