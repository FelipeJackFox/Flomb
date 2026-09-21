# ¿Sumar o promediar los olfateos? (prefijado 21 sept 2026, antes de evaluar)

El 85% de las muertes de la mosca son adivinanzas forzadas, y con la misma información una heurística que PROMEDIA adivina mejor que un aprendiz que SUMA. Hoy el valor de una casilla es la suma de las respuestas a sus olfateos,
así que una casilla con muchas vecinas abiertas pesa más solo por tener más. Cambio: valor = media de sus olfateos (normalización divisiva; no añade información). El patrón Kenyon con el que aprende es también la media.
Pilotos dev (benchmarks/fly-aggregate-pilot): suma + compuerta aprendida 71.9% en 9×9; media + compuerta aprendida 76–81% en 5 de 6 semillas y una semilla con la compuerta desbocada (θ = +3.9, marca todo, 12%);
media + reflejo FIJO de 3%: 33.7% con miles de marcas falsas (al promediar cambia la escala de cada olfateo y el 3% puesto a mano deja de servir: solo la compuerta aprendida se adapta); media + compuerta con tope θ ≤ 0
("no le tengas pavor a lo que crees más seguro que peligroso"): 78.6%, 8 de 8 semillas sanas.
Formal: olfateos crudos, no letal, densidad variable, 20k partidas, 10 semillas, unidad = semilla; evaluación letal con inicio en cascada en layouts reservados nuevos: 500 de 9×9/12, 250 de 7×7/7, 150 de 16×16/40.
Brazos: sum (configuración oficial: suma + compuerta aprendida, tope 4) · sum-cap (suma + tope 0, para aislar el efecto del tope) · mean-cap (media + tope 0). Política a mano en los mismos layouts.
Principal: mean-cap − sum en 9×9, pareado por semilla. Secundarios: 16×16, 7×7, sum-cap − sum, semillas desbocadas, marcas falsas, θ final.
