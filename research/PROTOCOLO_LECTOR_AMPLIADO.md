# Lector de contexto ampliado

Hipótesis: el lectorRF5 limita integración de información espacial de actividad. Cambiar segunda convdilation1→3,padding1→3,amplía campo5×5→9×9 sin cambiar12,769parámetros. No equivale a observar todo7×7 desde cadaesquina; encoder/cerebro ya mezclan información fuera delcampo dellector. Test porgradiente comprueba campos exactos y cuenta parámetros.

Trespares desde bestDAggeradaptado02/03/04. Control local vsampliado, mismos pesos/Adam/RNG iniciales, 1500updates adicionales,batch64 mezcla32originalesbalanceados+32DAggerheredado, exactamente mismos draws;encoder/cerebrocongelados. Sin nuevastrayectorias ni etiquetas. Todoslos padresseleccionados después1500 tenían disponibleelbuffer deterceraronda; usarlo completo. Dilation cambiafunción inicial aun conigualespesos y losmomentos heredados tienen otro contexto, limitaciónexperimental explícita.

Mínimapérdida originalholdout0/250/…/1500,empatemástemprano,sello antes de500layoutsnuevoscompartidos7×7/7minas. Evaluarbaseline/local/wide. Principal wide−local;secundariowide−baseline, bootstrap pareado portableroentre3semillas,un cerebro500compartidos,no1500independientes. Excluiraperturasautomáticas;maestronoactúa. Preservar originales,agenteservido intacto. Runner `experiments/train_wide_reader.py`, salida `runs/wide-reader-001`.
