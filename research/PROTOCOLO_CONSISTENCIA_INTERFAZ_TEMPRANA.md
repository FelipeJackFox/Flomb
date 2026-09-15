# Repetición de adaptación a un ciclo

Dos lectores preexistentes 20261104 y 20261105, desde propagation-time-SEED/latest-early.pt, encoder común deep-coverage/latest-autonomous.pt. 750 updates por brazo, mismo Adam/RNG inicial, head .001 y encoder .0001 nuevo solo en joint; grafo fijo. Mismo protocolo de piloto, preflight y mezcla 16 por fuente. Sin escoger checkpoints con resultados de test.

750 layouts nuevos por réplica (500 de9×9/12 y250 de7×7/7), disjuntos entre réplicas e histórico. Principal media joint-control en9 de las dos repeticiones; bootstrap pareado por layout dentro de semilla, piloto excluido. Secundario retención7. No nuevos cerebros, ni anatomía causal. Preservar fuentes y agente servido. Presupuesto total3000 updates,4500 episodios de evaluación contando baseline/control/joint; ejecución secuencial local.
