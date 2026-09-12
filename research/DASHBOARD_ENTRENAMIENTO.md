# Dashboard y métricas · 12 septiembre 2026

Abrir http://127.0.0.1:8765/dashboard.html. Chart.js 4.5.1 (MIT) está distribuido localmente con su licencia, sin depender de CDN. La página incluye 32 paneles, filtros por corrida y tamaño/minas, ventana temporal, tooltips, leyendas que permiten ocultar series, exportación JSON y actualización cada60segundos cuando está visible.

La vista por tamaño mezcla densidades: usar filtro exacto para concluir saturación. Evaluación periódica sin maestro/exploración separada del entrenamiento. La tabla muestra intervalos Wilson95% excluyendo aperturas ganadas. Auditoría del checkpoint33792 en panel separado: sus semillas/protocolo no se mezclan con las curvas de otras evaluaciones.

Las métricas de juego están descritas en METRICAS_BUSCAMINAS.md y los nuevos entrenadores en ../experiments/INSTRUMENTATION.md. Ausencias se muestran como tales, nunca como cero. Frecuencia de frontera/borde y mínimo riesgo inmediato no implican estrategia óptima. El score neutral50/50 asigna0.5 a ambos desenlaces certificados, por separado del win rate. Solo episodios medidos completos contribuyen al score.

Los cambios manuales de mezcla usan6estratos exactos:5/3,7/7,9/10,12/24,16/40,16/56. Se normalizan pesos, guardan solicitudes atómicas auditadas y se confirma la revisión aplicada al comenzar un episodio. El calendario original mantiene tamaños/densidades variables. No se alteraron los splits de la corrida antigua. Su proceso cargado no admite este control; las nuevas invocaciones sí.

Verificación: suite Python49pruebas; prueba del conectoma completo híbrido0→2→4 con reanudación, instrumentación, evaluaciones y adopción de mezcla enep3; API rechaza paths externos, controles incompatibles, mezcla nula y origen externo. El dashboard cuenta con lectura y selección WebMCP, comprobadas contra datos reales. No se realizó inspección visual de navegador en este turno.

La copia scene/dist/data/dashboard-snapshot.json permite servir la página como contenido estático, con controles deshabilitados. La API de escritura permanece solo en loopback. No está publicada todavía en LEIA: para control remoto se requiere integrar el backend autenticado de LEIA. No se ha creado un alojamiento separado.
