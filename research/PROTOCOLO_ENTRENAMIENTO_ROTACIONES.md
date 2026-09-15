# Aprender con orientaciones variadas

Encoder piloto early-interface-001/latest-joint ygrafo fijo1ciclo. Dos lectores nuevos idénticos12769parámetros,seed20261108,Adam.001nuevo,3000updatesfijos,batch64,16por4fuentes históricas. Mismos índices base ypresupuesto;brazo aumentado orientaciónuniforme0/90/180/270porposición,controloriginal. Rotar observaciones,legalidad ysafe-labels ensubtableroNxN,conservarpadding/contexto. RNGrotaciones separado de muestreo. Caché4orientaciones recomputada,controlusa k0;no modificar datasets históricos.

750layouts nuevos compartidos5009/2507,principal9aumentado-control,secundario7. Ambosuna inferencia porjugada;sin ensemble,maestroacciones ni selecciónendpointtest. Bootstrap pareado10000. Presupuesto6000updates/1500episodios,preprocesamiento4orientaciones por estado. Checkpointparent no se escoge por test. Fuentes/agente servido intactos. Validar alineación de labels/legalidad,rotacióninversa,conteos ypadding antes de lanzar.
