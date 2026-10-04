# Réplica para el número duro (prefijado 4 oct 2026, antes de evaluar)

El 81.3% [78.8, 83.8] de runs/fly-aggregate-002 salió de una sola corrida de 10 semillas y de un conjunto de 500 tableros que resultó más fácil que otros (la política a mano sacó 84.8% ahí y 78.8–80.8% en otros).
Réplica sin cambiar nada de la configuración de trabajo: olfateos crudos (N,k,m), entrenamiento no letal, densidad variable 8–25%, compuerta de marca aprendida con tope θ ≤ 0, cada olfateo condicionado por su cuenta y
decisión promediada; 20,000 partidas de entrenamiento alternando 7×7 y 9×9.
- 30 semillas nuevas (20–49), nunca usadas en ningún piloto ni corrida.
- Evaluación letal con inicio en cascada en layouts reservados nuevos: 2,000 de 9×9/12 minas, 500 de 7×7/7, 300 de 16×16/40. Política a mano sobre los mismos tableros.
- Unidad de análisis = semilla. Se reporta media, desviación, IC95 t, mínimo y máximo por semilla, moscas hundidas (definidas de antemano como < 50% en 9×9), marcas falsas, y la diferencia con la política a mano.
Este es el número que se cita. No hay brazo de comparación: no se prueba ninguna hipótesis nueva, solo se mide.
