# Inicio con apertura garantizada y tableros grandes (prefijado 20 sept 2026, antes de evaluar)

Observación de Felipe al ver partidas en el visor: el inicio actual (una casilla central segura) es injusto. Medido: en 9×9/12 minas esa apertura destapa UNA sola casilla con número en el 73.9% de las partidas
(mediana de casillas abiertas = 1), lo que obliga a adivinar a ciegas de entrada. El estándar moderno (p. ej. minesweeper.now, Windows Vista en adelante) garantiza que la primera casilla Y sus ocho vecinas
están libres de minas, de modo que la partida abre en cascada. Implementado como Minesweeper(..., zero_start=True): mediana 40 casillas abiertas en 9×9, mínimo 9. El valor por defecto no cambia (corridas previas reproducibles).
Base: mejor receta legítima (olfateos crudos N,k,m; entrenamiento no letal; densidad variable 8–25%; reflejo de marca 3%). 8 semillas por brazo, unidad = semilla.
Brazos de ENTRENAMIENTO: old (inicio viejo, 7×7 y 9×9, 20k partidas) · zero (inicio con cascada, 7×7 y 9×9, 20k) · zero-big (inicio con cascada, 9×9 y 16×16, 10k partidas ≈ mismo número de pisadas).
Conjuntos de EVALUACIÓN (letal, layouts reservados nuevos): 9×9/12 inicio viejo (400) · 9×9/12 con cascada (400) · 16×16/40 con cascada (150) · 30×30/140 con cascada (30; "casi infinito").
Métricas: % victorias; en tableros grandes además fracción del tablero despejada antes de morir y % de pisadas seguras, porque ganar 30×30 exige cientos de decisiones seguidas. Política a mano en cada conjunto como techo.
Preguntas: (1) ¿cuánto del déficit era del inicio injusto? (evaluar con cascada vs sin ella, misma mosca); (2) ¿entrenar con cascada enseña algo distinto? (zero vs old, ambos evaluados con cascada);
(3) ¿entrenar en tableros grandes ayuda o transfiere? (zero-big vs zero en 9, 16 y 30). Nota: los sentidos son locales (3×3 alrededor de cada pista), así que un tablero infinito solo cambia la proporción de bordes y la duración; no hay nada que la mosca perciba como "tamaño".
