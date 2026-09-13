# Decoder espacial: piloto autorizado

Objetivo: medir si dar contexto neuronal vecino al lector mejora decisiones y victorias autónomas, sin entrenar nuevamente el cerebro.

- Conservar encoder, ganancias neuronales, conexiones y mapa de retina del checkpoint retina_plastic.
- Comparar original, lector puntual nuevo y lector espacial nuevo. Los dos nuevos tienen ~12.8k parámetros y mismo presupuesto de entrenamiento.
- Cachear actividad de las 10,000 posiciones reales y 1,000 de validación anteriores. Mismos minibatches, 1,000 updates/batch64/lr0.001/semilla20261002. Elegir checkpoint por validación.
- Evaluar las tres políticas en las mismas 500 partidas nuevas, sin maestro; 250 por dificultad 5×5/3 minas y 7×7/7 minas.
- Separar aperturas automáticas, oportunidades seguras, errores en minas demostrables y resultados discordantes por partida.
- La comparación permite valorar este lector en este checkpoint; una sola semilla no demuestra superioridad general ni ventaja biológica. No publicar ni sustituir automáticamente el agente servido.

Fuentes y resultados: experiments/spatial_decoder.py, experiments/train_spatial_decoder.py, runs/spatial-decoder-001/. El checkpoint original y el híbrido son de solo lectura.

Ajuste antes de entrenar/evaluar: el universo 5×5 contiene C(24,3)=2,024 layouts y 1,897 ya aparecían en datasets previos. Se evaluarán los 127 inéditos restantes y 250 de 7×7 (377 total). Se conservó la caché y se acotó explícitamente la búsqueda; ningún resultado de evaluación influyó en este ajuste.
