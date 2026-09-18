# Fusión aprendida de orientaciones (prefijado 18 sept 2026, antes de lanzar)

Motivo: promediar logits de 4 rotaciones es la única mejora replicada (+16.9 pp en 9×9). Enseñar al lector a girar
(aumentación, destilación) no la trasladó a una sola inferencia, y los reflejos no añadieron nada. La ganancia nace de que el
cerebro fijo NO es equivariante: A(R^k x) ≠ R^k A(x), así que cuatro pasadas dan cuatro lecturas distintas del mismo tablero.
Un lector equivariante sobre UNA pasada no puede recuperar eso. La pregunta útil es si combinar las cuatro lecturas con un
lector aprendido rinde más que el promedio de logits, al mismo coste de inferencia.

Encoder piloto early-interface-001/latest-joint y grafo fijos, 1 ciclo. Tres lectores nuevos, seed 20261110, Adam .001,
3000 updates fijos, batch 64 (16 por cada una de las 4 fuentes históricas), mismos draws de posiciones en los tres brazos.

- mean: ActivityHead 12769 parámetros entrenado con orientación original; en partida promedia logits de 4 rotaciones (RotationPolicy).
- fusion: FusionHead 12739 parámetros; entrada = 4 actividades giradas de vuelta al marco del tablero (40 canales) + contexto.
- fusion_aug: igual que fusion, pero cada posición se presenta en un giro uniforme 0/90/180/270 tomado de la misma caché
  (S(R^j x)_k = R^j S(x)_{(k+j)%4}; sin pasadas de cerebro extra). RNG de giros separado del muestreo.

Los tres brazos: 4 pasadas de cerebro por jugada. 750 layouts nuevos compartidos (500 de 9×9/12 minas, 250 de 7×7/7 minas).
Principal: 9×9 fusion_aug − mean. Secundarios: 9×9 fusion − mean y ambos en 7×7. Bootstrap pareado 10000. Endpoint fijo 3000,
sin selección por test, sellado antes de evaluar. Fuentes, checkpoints históricos y agente servido intactos. Validado antes de
lanzar (experiments/test_orientation_fusion.py): rotación idéntica a la de referencia, padding intacto, inversa exacta, giro desde
caché idéntico a recomputar el tablero girado, y política de partida idéntica al stack de entrenamiento.

Límite: una semilla de lector y un cerebro; un resultado favorable exige réplica con los modelos 04/05 antes de creerlo.
