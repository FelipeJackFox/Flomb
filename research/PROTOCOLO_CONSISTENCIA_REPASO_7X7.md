# Replicación del repaso7×7

El piloto02 obtuvo57/250vs46/250control en7 (+4.4pp,IC[.4,8.4]),y11/500vs13/500en9(-.4pp,IC[-2,1.2]). Se repite sin cambiar método ni presupuesto en semillas03/04 desde sus propios padres9latest2250.

Cada par hereda pesos/Adam/RNG y continúa1500updates:control32originales+32experiencia9fija,repaso16originales+16DAgger7heredada+32idéntica9. Último1500fijo,no selección por test. Encoder/grafo y mapeo fijos. Caches originales y experiencia propia de cada semilla,verificadas con etiquetas/contexto/máscaras exactas y actividadrtol1e-5/atol2e-6.

Principal media7repaso−control de DOS nuevas réplicas;secundarias coste9 yvsparent9. El piloto02no se reutiliza en la principal. Cada réplica reserva750testnuevos propios(2507/5009),disjuntos entre sí y de históricos;dentro de cada réplica tres brazos comparten layouts. Bootstrap estratificado por semilla ypareado por tablero,condicional a dos modelos sobre un cerebro,1500layouts totales. No afirmar equivalencia/noinferioridad9 ante un intervalo inconcluso.

Runner experiments/repeat_seven_rehearsal.py,coordinador runs/seven-rehearsal-consistency-001,hijos runs/seven-rehearsal-20261003/4. No cambios al agente servido ni nuevos datos de entrenamiento.
