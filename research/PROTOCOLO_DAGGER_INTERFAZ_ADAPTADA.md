# DAgger sobre interfaz adaptada

Autorización continua del usuario. Piloto semilla20261002 desde `runs/joint-extended-20261002/best-joint.pt` elegido2750. Hipótesis: la experiencia autónoma nueva aporta más que repetir datos fijos una vez adaptada la interfaz. DAgger ya ayudó al lector con la interfaz anterior; esta combinación aún no se ha probado.

Congelar encoder adaptado y cerebro completo; entrenar solo lector. Conservar Adam del lector exactamente al retirar el grupo congelado del encoder y restaurar RNG inicial igual en ambos brazos. Tres rondas300partidas nuevas autónomas+750updates por brazo(2250totales). Control64ejemplos originales equilibrados;DAgger32originales+32uniformes del agregado nuevo. Maestro etiqueta acciones certificadas seguras,no decide acciones(beta0). Mismo optimizador/clipping/loss;sin máscara del maestro al jugar. Cache recomputada para el encoder adaptado; no usar actividad vieja.

10,000originales/1,000holdout;mínima pérdida cada250 incluyendoinicial.900layouts colección y500finales nuevos disjuntos entre sí y de listas/datasets previos. Sellar selección antes de500partidas finales baseline/control/DAgger,excluir aperturas automáticas. Principal DAgger−control;secundario DAgger−baseline;una semilla y un cerebro,no consistencia todavía. Preservar fuentes,historial,checkpoints;no modificar agente servido ni cómputo externo.
