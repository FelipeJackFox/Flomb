# Diseño acordado

Prioridad actual: entrenamiento de Buscaminas; visualización después.

## Escena

Mosca sentada frente a una computadora y usando un mouse diminuto. La política
elige la casilla y una animación lleva el cursor a ella; el agente no aprende
locomoción ni control físico de un mouse. Vista en tres cuartos y tablero ampliado.

Otra zona de la pantalla muestra el cerebro 3D interactivo, actividad neuronal y
conexiones, sincronizados con las decisiones. Diferenciar actividad instantánea
de cambios de pesos por entrenamiento; no presentar conexiones activas como nuevas.
Permitir filtros por región y selección de conexiones para evitar saturación visual.

La versión inicial del entrenador es una red de tasas: su actividad podrá mostrarse
como intensidad continua. No debe etiquetarse como spikes ni como tiempo biológico.

El render será independiente del entrenamiento; las partidas de evaluación pueden
reproducirse a velocidad humana. No se necesita renderizar cada episodio de entrenamiento.
