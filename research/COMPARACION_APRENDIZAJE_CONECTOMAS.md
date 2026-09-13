# Qué está aprendiendo realmente cada proyecto y qué limita nuestro modelo

Investigación y auditoría del código: 12 de septiembre de 2026. Las conclusiones sobre proyectos externos describen lo publicado; no son reproducciones independientes.

**Conclusión:** nuestra restricción de plasticidad es una hipótesis seria para explicar el mal aprendizaje. Conservar todo el grafo no equivale a disponer de un modelo tan expresivo como otros trabajos. Sin embargo, añadir spikes o mayor detalle biológico no garantiza resolverlo: existen resultados con neuronas simplificadas y demostraciones biológicas sin aprendizaje útil demostrado.

## Videos frente a evidencia

| Proyecto | Qué permite concluir la fuente primaria |
|---|---|
| [DOOMFLY](https://fly-brain-doom.awormuth.chatgpt.site/learning) | Su laboratorio declara que el aprendizaje de supervivencia sigue sin demostrarse. Cambios sinápticos y respuesta a estímulos no establecen mejor desempeño. |
| [Stonkfly](https://github.com/nftechie/stonkfly/blob/main/docs/model.md) | Conecta actividad neuronal a propuestas de compra/venta mediante una lectura fija y plasticidad experimental. Declara que no ha demostrado aprendizaje rentable ni mejora de estrategia. |
| [Fly Brain Minecraft](https://github.com/blendi-remade/fly-brain-minecraft) | Simula actividad y conductas evocadas. Su README distingue circuitos, ganancias manuales, una máquina de estados de vuelo y un reflejo programado para acercarse a comida. No presenta una evaluación de aprendizaje de estrategias de Minecraft. Esta revisión corresponde a ese repositorio, no a todos los videos de Minecraft. |
| [Flybody](https://github.com/TuragaLab/flybody) | Es una plataforma de simulación corporal. Ver una mosca caminar con un controlador aprendido no demuestra por sí solo que el controlador utilice el conectoma. |

Beat Saber: no pude verificar un repositorio y evaluación autónoma de la demostración viral de Lyra Bubbles. Por ello no la uso como evidencia técnica de una receta de entrenamiento exitosa ni concluyo que haya generalizado a canciones nuevas.

## Trabajos con métodos y evaluaciones publicados

**[FlyGM](https://arxiv.org/html/2602.17997v1)**, preprint de locomoción: utiliza FlyWire femenino, mantiene el operador sináptico fijo y aumenta la capacidad de cada nodo mediante estados de 32 canales, descriptores entrenables y una actualización MLP compartida. Aprende interfaces de entrada y salida. Primero imita un experto y después ajusta con PPO; publica uso de A100 de 80 GB. Es una alternativa relevante a aprender cada sinapsis individualmente. Su resultado locomotor no demuestra razonamiento de Buscaminas.

**[FLYNN](https://arxiv.org/html/2607.00025v1)**, preprint de navegación: usa neuronas leaky-integrator con tanh, pesos, sesgos y tasas de fuga entrenables. DAgger agrega 500 episodios por iteración, durante cuatro iteraciones. Equilibra ejemplos de inicio, precolisión, colisión, avance y giro. El maestro usa información privilegiada para planificar; el alumno recibe visión, colisiones y dirección relativa del objetivo. Publica evaluaciones con cambios de entorno y pérdida de sensores. Usa FlyWire femenino y conexiones con al menos cinco contactos; no es el mismo grafo MaleCNS que nosotros.

La lección transferible es combinar capacidad entrenable, interfaces adecuadas, demostraciones y cobertura deliberada de situaciones. Los trabajos no aíslan una sola causa universal del éxito.

## Auditoría de nuestra implementación

Código relevante: `prepare_data.py`, `brain.py`, `experiments/backbone.py` y `experiments/qr_core.py`.

- Retenemos 166,700 neuronas y 25,582,938 conexiones. La limitación principal identificada no es haber seleccionado un pequeño subcircuito.
- Cada neurona tiene un único valor de actividad. La propagación usa tanh durante tres ciclos.
- Dentro del grafo solo aprende una ganancia por neurona: todas sus conexiones salientes se escalan juntas. No puede reforzar una salida y debilitar otra independientemente.
- La entrada emplea proyecciones aleatorias fijas. La lectura selecciona 1,024 neuronas; la cabeza QR es lineal dueling.
- No hay sesgos internos entrenables, actualización neuronal MLP ni dinámica de fuga aprendida. Cada decisión reinicia el estado a partir de la observación.
- Los pesos se derivan de contactos y supuestos de signo, con normalización por entrada. No son pesos funcionales medidos de una mosca viva ni una política preentrenada.

Por tanto, “entrenamos el cerebro completo” debe entenderse como actualizar ganancias internas a través de todo el grafo, no como entrenar libremente sus millones de conexiones. Esta distinción debía haberse hecho explícita al comparar arquitecturas.

Posibles cuellos de botella: entrada poco apropiada para relaciones entre casillas, transformación interna restringida y pérdida de señal tras agregaciones. Son hipótesis, no causas demostradas. La ausencia de memoria entre decisiones no basta como explicación: el tablero visible ya conserva información de movimientos anteriores.

El ensayo local 3×3 fue un promedio fijo, no una CNN aprendida. Su falta de ventaja no descarta un codificador entrenable. Los resultados están en `RESULTADO_ENTRADA_ESPACIAL.md`.

## Experimento que recomiendo, antes de otra corrida larga

1. Construir un conjunto supervisado más amplio, separado por tablero, con decisiones seguras certificadas usando únicamente pistas visibles. Balancear reglas elementales, relaciones entre restricciones y estados donde hay que estimar riesgo. Evaluar el conjunto de acciones válidas, evitando penalizar elegir otra casilla igualmente segura.
2. Entrenar un control CNN pequeño con los mismos datos y presupuesto. Sirve para comprobar si datos, etiquetas y objetivo permiten generalizar; no sustituye el objetivo del cerebro completo.
3. Comparar el modelo actual con una variante del conectoma que aprenda entrada y transformación interna. Empezar con pocos canales y medir costo; aumentar canales o habilitar pesos sinápticos son alternativas que deben evaluarse por separado. Conservar los nodos y conexiones retenidos.
4. Comparar con un grafo reconfigurado que preserve grados y distribución de pesos, con interfaces y capacidad equivalentes. Así se distingue aprender con esa arquitectura de obtener una ventaja por la conectividad biológica.
5. Exigir mejora en tableros reservados y partidas autónomas antes de volver a DAgger + QR-DQN. Reportar por dificultad, oportunidades seguras desaprovechadas y riesgo evitable; mantener separado el resultado real del ajuste por azar inevitable.

Una arquitectura con varios canales encarece las operaciones dispersas. La A100 del artículo no predice nuestro tiempo en el M4. Debemos medir memoria, actualizaciones por segundo y mejora por minuto con pilotos emparejados, no prometer aceleración a partir de RAM libre.

Esta investigación no modifica checkpoints ni reanuda entrenamientos.
