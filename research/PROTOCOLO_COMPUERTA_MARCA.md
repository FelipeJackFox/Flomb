# Frente 5: la compuerta de marcar, aprendida (prefijado 21 sept 2026, antes de evaluar)

En runs/fly-marking-001 el umbral aprendido se quedó en θ ≈ −6.1 (0.2% de seguridad) y perdía 16 pp contra el 3% puesto a mano (θ = −3.48). Diagnóstico en pilotos dev (benchmarks/fly-gate-pilot*):
con τ = 1 (compuerta blanda) θ converge a ≈ −6.3 DESDE AMBOS LADOS (partiendo de −8 y de −1): no era estancamiento sino un equilibrio. La compuerta blanda marca al azar casillas seguras cerca del umbral mientras explora,
eso se castiga después, y el umbral se retira hasta donde ya no hay olores cerca. Con compuerta más nítida (τ = 0.25) partiendo de −8, θ sube a entre −3.1 y −5.1 y la mosca juega casi igual que con el reflejo de 3%
sobre el mismo cerebro (67.3 vs 69.7), con cero marcas falsas. Partiendo de −1 (marcando de más) es inestable: en algunas semillas θ se dispara a +4 y marca todo.
Evaluación formal. Base: olfateos crudos, no letal, densidad variable, 20k partidas, 10 semillas, unidad = semilla; evaluación letal con inicio en cascada en layouts reservados nuevos (500 de 9×9/12, 250 de 7×7/7).
Brazos: gate (θ0 = −8, τ = 0.25, tasa 0.02 que decae 1/(1+g/3000), calor −3 / azúcar +1, traza 0.9) · reflex (3% fijo) · sham (misma compuerta, pero la dopamina que recibe es aleatoria con la misma frecuencia)
· never (no marca jamás: θ = −30, sin aprender; esta vez sí es un control limpio).
Principal: gate − reflex en 9×9 (¿el umbral aprendido iguala al puesto a mano? margen de equivalencia que considero aceptable: ±5 pp). Secundarios: gate − sham, gate − never, θ final y dispersión, marcas falsas, semillas que se disparan.
