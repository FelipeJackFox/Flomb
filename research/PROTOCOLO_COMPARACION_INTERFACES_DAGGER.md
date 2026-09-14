# Aporte de adaptación previa a DAgger

Autorizacióncontinua. Trasconfirmar DAggeradaptado sobrecontrol, comparar pipelineadaptado vs noadaptado con presupuestosprevios deactualizacionesiguales. No es ablación aisladaencoder: adaptación previa cambia encoder/lector/Adam/RNG,seleccionadospor misma regla; cuantifica valor deesa etapa delpipeline.

Tressemillas20261002/3/4. AdaptadosDAgger existentes preservados. Nuevos noadaptados parten de `runs/joint-extended-SEED/best-control.pt`,control que recibió igualpresupuesto3000 antesdeselección; encoderoriginal fijo. Cadauno ejecuta mismoDAgger3rondas300partidas+750updates(2250),mismas tasas,Adam/RNGheredados,etiquetassafe50%original/50%agregado. Reutilizar exactamente los900layouts decolección de su contraparte adaptada; trayectorias y ejemplos resultantes pueden variar. No confundir esos layouts históricos autorizados paraentrenar con testnuevo.

Nuevo benchmark500layouts7×7/7minas disjunto detodoloanterior; sellar seis candidatos antesdeevaluar. Principal media adaptado−noadaptado pareadaportablero entre3semillas;ICbootstrapcondicional,un cerebro500tableroscompartidos,no1500independientes. Teacheracciones0,encoder/grafofijosduranteDAgger. Mismolectorcapacidadypresupuestodeupdates; distinto costetiempo histórico.No reemplazar servido.

Runner`experiments/compare_dagger_interfaces.py`,coordinador`runs/dagger-interface-comparison-001`,nuevos`runs/unadapted-dagger-SEED`.
