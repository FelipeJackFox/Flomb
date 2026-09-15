# Control de representación con datos actuales9×9

Masa segura no mejora9frenteCE(24vs28/500,delta−.8pp IC[-2.8,1.2]). Tras adaptaciónencoder,datos9yrepaso,comparar qué permite aprender el mismolector con pistas crudas frenteactividad. La antigua comparación de entradas evaluaba7trasdatos5/7;esta prueba introduce la experiencia9 actual yencoderadaptado9. No es repetir idéntico experimento.

Doslectores ActivityHead(True),12,769parámetros,misma inicialización fresca semilla20261103,mismoAdamnuevo. Entrada raw:10canalesonehot de pistas/oculto ypaddingceros;entrada brain:10canalesdeactividad trasencoder9/grafofijos. Ambosañaden contexto públicotamaño/densidad. Nadie recibe minas ocultas. Encoderfijo de deep-coverage-001/latest-autonomous.pt. El brazo raw es control de diagnóstico,NOse le atribuye usar el cerebro.

3000updatesporbrazo,B64,mismosíndices16original+16DAgger7+16DAgger9anterior+16recienteautónoma;CEuniforme,Adam.001,clip5. Igualpresupuestodellector,no igualdaddecómputohistóricototal(encoder/grafo tienen entrenamiento previo). No heredarlectores para evitar iniciar unbrazo conpesos ajustados a otraentrada. Último3000fijoantesdetest,sin escoger según resultados.

Test750layouts nuevoscompartidos5009/2507. Principal9raw−brain,secundario7. Una inicialización/un cerebro;no prueba imposibilidad biológica ni una ventaja anatómica. Recomputarfeatures yverificar máscaras/contextos/etiquetasidénticosentreentradas. Preservaragenteservido ytodoslosmodelos. Runner experiments/compare_nine_representation.py,run runs/nine-representation-001,reportero experiments/report_nine_representation.py.
