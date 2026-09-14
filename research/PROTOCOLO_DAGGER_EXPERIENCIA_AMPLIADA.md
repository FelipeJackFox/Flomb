# Ampliación de experiencia DAgger

Tres semillas del lector (20261002/20261003/20261004), cada una parte de su mejor checkpoint DAgger elegido por validación. Se restauran pesos, Adam y RNG; cerebro, encoder y mapa óptico permanecen congelados. Se conserva la experiencia recogida en las primeras tres rondas de cada semilla.

Tres rondas adicionales de 600 partidas nuevas por semilla: 1,800 layouts de colección compartidos entre lectores para controlar diferencias de tableros, con trayectorias propias. Maestro etiqueta solo estados con alguna jugada certificada segura; el lector ejecuta todos los clics y termina al morir. Cada ronda agrega 750 updates, batch 64, Adam lr 0.001, clipping 5.

Tres brazos por semilla:
- Baseline: DAgger inicial de esta fase, intacto.
- Control: mismos pesos/Adam/RNG iniciales y 2,250 updates adicionales, mezcla 32 posiciones originales equilibradas + 32 posiciones del buffer DAgger heredado fijo.
- Ampliado: mismo presupuesto; mezcla 32 posiciones originales + 32 del buffer agregado (heredado y nuevas rondas).

Así se prueba agregar experiencia frente a continuar sobre toda la experiencia ya disponible, no frente a un control privado de datos DAgger anteriores. Selección solo por mínimo de pérdida en las 1,000 posiciones originales de validación, cada 250 updates, incluyendo baseline. La colección usa los pesos actuales; la selección final puede regresar a un punto previo.

Se reservan antes de entrenar 500 layouts finales 7×7/7 minas y 1,800 layouts de colección, sin solapamiento entre sí ni con datasets, benchmarks o colección previos. Los nueve modelos se evalúan en las mismas 500 partidas sin maestro. Apertura central segura idéntica. Reportar victorias y errores evitables; bootstrap emparejado por tableros condicionado a los tres lectores, no 1,500 tableros independientes.

Actividad cacheada del cerebro congelado, compartida entre semillas solo por observación pública/contexto. Se guardan actividad y etiquetas por ronda, datasets agregados, mejores/últimos checkpoints con Adam/RNG, fuentes y hashes originales. Cinco pruebas del colector/memo/lector antes de iniciar; auditoría independiente de splits, etiquetas, caché y checkpoints al terminar. No reemplazar agente servido ni cambiar el cerebro en esta fase.
