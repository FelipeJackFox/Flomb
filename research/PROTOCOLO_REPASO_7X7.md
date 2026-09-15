# Repaso de experiencia7×7 durante continuación9×9

Las dos nuevas réplicas confirman mejora9 frentecontrol (+2.1pp,IC[1.2,3.1]) pero pierden4pp en7 frente sus baselines (IC[-7.6,-.4]). La diferencia7 contra controles adicionales es inconclusa: no atribuir toda la pérdida a introducir9. El entrenamiento usó ejemplos originales5/7, pero dejó fuera la experiencia DAgger7 heredada que había mejorado esas políticas.

Hipótesis: incorporar esa experiencia antigua ayuda a retener7 frente a continuar solo con originales y experiencia9. Piloto02 desde `runs/nine-dagger-001/latest-dagger.pt`, paso2250, copiando pesos/Adam/RNG a ambos brazos.1500updates adicionales,último fijo en ambos (total3750). Control32originales+32experiencia9fija;repaso16originales+16DAgger7heredada+32mismaexperiencia9. Generar idénticos índices para los tres depósitos en ambos brazos, compartir primeros16 originales y32de9;única sustitución son los otros16originales por16DAgger7. B64,Adam heredado,clip5.

No recolectar nuevas partidas de entrenamiento ni reentrenarencoder/grafo. Datos9fijos4533estados delpiloto;datos7heredados de adapted-dagger-001 ycache wide-reader-001. Verificar igualdad delencoder y recomputar muestras de lascaches antes de usarlas. Los estados nuevos se procesan con el mapeo extendido comprobado.

Principal: repaso−control en250test7nuevos;secundarias500test9,comparación conpadre9preservado y ratios de decisiones seguras.750layouts nuevos disjuntos de todos los anteriores,unlector/un cerebro. Mostrar el coste en9 aunque7mejore;no declarar no-inferioridad formal ni éxito por pérdida. Último1500local fijado antes de evaluar,no elegir por estas partidas.

Runner experiments/train_seven_rehearsal.py,run runs/seven-rehearsal-001,informe experiments/report_seven_rehearsal.py. Mantener agenteservido intacto.

Preflight: misma actividad float32 puede variar ligeramente al cambiar el tamaño de batch de inferencia. Verificar pesos de encoder y etiquetas/contexto/máscaras exactamente; comparar actividad con rtol1e-5,atol2e-6 y registrar máximo error. Primer intento estricto falló con máximo1.01e-6 y quedó preservado en seven-rehearsal-preflight-001 sin entrenamiento. La repetición reserva test nuevo.
