# Cómo entrenan tareas difíciles con conectomas de mosca

Revisión: 12 de septiembre de 2026. Lectura de fuentes; no reproducción independiente de resultados. No todos usan MaleCNS: varios trabajos anteriores usan FlyWire/FAFB de una hembra. Un cuerpo Flybody tampoco implica por sí mismo que el controlador sea un conectoma.

| Proyecto | Método documentado | Qué podemos aprovechar |
|---|---|---|
| [FlyGM, §3.2](https://arxiv.org/html/2602.17997v1) | Preentrena una política estructurada por el conectoma con demostraciones de un experto MLP, imitando su distribución de acciones con KL y MSE decreciente. Después usa PPO, crítico MLP, entropía y rollouts paralelos. | Inicializar con decisiones útiles antes de depender de premios escasos. Es un preprint de locomoción; no demuestra rendimiento en Buscaminas ni en MaleCNS. |
| [FLYNN, §II-D](https://arxiv.org/html/2607.00025v1) | DAgger: agrega datos nuevos y reduce la proporción de control del maestro. El maestro planifica con información privilegiada; el alumno recibe sensores. Balancea tipos de conducta y aleatoriza escenarios. Usa FAFB v783 y entrena pesos, sesgos y fugas. | Consultar al solver en los estados que visita nuestra política, incluso después de sus errores; entrenar sobre esa distribución. No hace falta dar posiciones ocultas de minas a nuestro maestro. |
| [Flybody, Nature 2025](https://www.nature.com/articles/s41586-025-09029-4) | RL con objetivos de imitación de trayectorias de moscas filmadas; controladores de bajo nivel dirigibles. El controlador es una red artificial y no una reconstrucción del conectoma. | Separar decisión del juego y animación motora. No necesitamos aprender a mover una pata para aprender qué casilla abrir. |
| [DOOMFLY, laboratorio del autor](https://fly-brain-doom.awormuth.chatgpt.site/learning) | Experimenta con plasticidad dependiente de dopamina en conexiones KC→MBON, estímulos por daño y controles congelados o con recompensa desordenada. | Copiar la disciplina de comparar contra controles. El autor declara que la mejora de supervivencia por aprendizaje sigue sin demostrarse. No tomar las luces ni los cambios de pesos como prueba. |
| [Fly Brain Minecraft, README](https://github.com/blendi-remade/fly-brain-minecraft/blob/main/README.md) | Simulación LIF con entradas sensoriales y lectura de poblaciones motoras. Distingue circuitos activados, ganancias elegidas a mano, reflejos y máquinas de estados. | Verificar qué parte aprende. Este mod no es evidencia de un agente que haya aprendido tareas generales de Minecraft. Hay varios proyectos diferentes con nombres parecidos. |

Para Beat Saber localicé cobertura y enlaces al autor, pero no pude verificar en esta revisión un repositorio y evaluación independiente de las ayudas del maestro. No uso ese video como evidencia de generalización a canciones desconocidas. Para cripto tampoco establecí un resultado de aprendizaje reproducible.

## Propuesta para nuestro Buscaminas

La corrida actual usa política estocástica y retorno episódico; ya recibe un tablero visible codificado, no píxeles. Su apertura central es segura, aunque no garantiza una región grande ni que exista una jugada deducible después. Aumentar episodios por sí solo no garantiza que aprenda las reglas.

1. **Imitación del solver visible.** Generar ejemplos de casillas demostrablemente seguras y de riesgo mínimo cuando sea necesario apostar. Entrenar probabilidad sobre el conjunto de acciones seguras, no imponer una única casilla cuando varias son equivalentes. Conservar separación entre deducción y estimación aproximada.
2. **DAgger.** Dejar actuar a la política y pedir al solver etiquetas para los tableros que realmente visita. Reducir gradualmente la ayuda, conservando etiquetas. El solver nunca entrega el mapa oculto de minas.
3. **RL después de imitación.** Comparar PPO con crítico por estado contra nuestro entrenamiento actual. Esto sería una corrida nueva, no una modificación silenciosa de la actual. Preservar los checkpoints de referencia.
4. **Dificultad basada en dominio.** Como experimento posterior al currículo ya autorizado: subir dificultad solo al superar un umbral en evaluación separada y mantener algo de práctica fácil para medir olvido. No cambiar ahora el decay a grandes de la corrida en curso.
5. **Evaluación que explique los fallos.** Medir victoria por tamaño/densidad, muertes teniendo una jugada segura deducible, decisiones forzadas a apostar, riesgo elegido y progreso. Usar múltiples semillas de entrenamiento y tableros de prueba nuevos. Comparar mismo presupuesto contra solver, red inicial, cerebro congelado y una red convencional; para atribuir ventaja a la topología, también grafo reordenado conservando grados.

Mi prioridad sería imitación + DAgger antes de comprar más cómputo. Es una propuesta respaldada por métodos de otros dominios, no una garantía de que nuestro modelo aprenda. No se inició otro entrenamiento ni se cambió el actual durante esta investigación.
