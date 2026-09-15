# Objetivo de escoger cualquier casilla segura

Cobertura corregida dio22/500vs28/500control9,delta−1.2pp IC[-3.2,.8];aumentóprofundidad de estados,pero no mejoró ejecución autónoma. Antes de más recolección,probar un cambio de objetivo con igualesdatos.

Pérdidaactual:entropíacruzadaconobjetivouniforme sobre todaslasaccionescertificadasseguras. Penaliza concentrarprobabilidadenunasola,inclusosiellaessegura. Alternativa: `logsumexp(logits legales)-logsumexp(logits seguros)` =−log(probabilidadtotaldelconjuntoseguro). No requiere repartiruniformemente dentrodelconjunto. Es unobjetivodistinto,no garantia demayorwinrate. Seguras sonlascertificadaspúblicamente,no toda casilla realmente libre.

Piloto02desde deep-coverage-001/latest-autonomous.pt:1500updatesadicionalesfijos,lectorúnicamente,mismos pesos/Adam/RNG/índices ybatches. ControlCEuniforme vsmasa segura. B64,16originales+16oldDAgger7+16oldDAgger9+16nuevaexperienciaautónomaheredada. Todasfeaturesrecomputadasparaencoder9actual,fijoigualquegrafo. No usar datos corregidos,nueva colección,auxiliares ni filtros de seguridad eninferencia.

Tests:equivalenciaCEcuandohayunasolaetiqueta,invarianciaa reparto internoaigualmasasegura,gradienteilegalcero,desplazamientologits yrechazodeetiquetasinválidas. Último1500prefijado,no selecciónporlossincomparable. Test5009+2507nuevos,principal9masa−control,secundarias7yvsparent;unlector/uncerebro. Informar ratioseguro y posiblescostes deconcentraracciones. Agente servido ycheckpoints intactos.

Runner experiments/train_safe_mass.py,run runs/safe-mass-001,reportero experiments/report_safe_mass.py.
