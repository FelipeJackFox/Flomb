# Actividad de dos tiempos

Hipótesis: los pasos1 y2 contienen información complementaria aunque2solo no mejore. Encoder fijo early-interface-001/latest-joint.pt y grafo fijo. Combinado concatena10canales actividad1+10actividad2;control concatena actividad1 dosveces. Ambos22entradas incluyendo2contexto,15649parámetros,lectores nuevos iguales seed20261107,Adam.001nuevo,3000updatesfijos,mismosdatos/draws16por4fuentes.

Caches separadas por tiempo yencoder;comprobar igualdad exacta primera mitad entre brazos ymitades delcontrol. No pistas directas,ni entrenar grafo.750testnuevos5009/2507,principal combinado-control9,secundario7,bootstrappareado10000;no seleccionar checkpoint por test.6000updates/1500episodios presupuesto local. Agente servido yfuentes intactos. No ventaja anatómica ni ciclos biológicos.
