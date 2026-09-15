# Adaptar codificador para lectura de1ciclo

Dosnuevasréplicas confirmaron1vs3ciclos:31/44vs15/18de5009,media+4.2pp IC[2.4,6].7media+3pp IC[-1.4,7.4]. Elencoder fuepreentrenadopara3;probar adaptaciónespecíficaa1sin cambiar grafo.

Pilotolector20261103(originaldelprimerpar,noelegirelmejorporvictorias) desde propagation-time-001/latest-early.pt yencoder deep-coverage-001/latest-autonomous.pt. Guardarbaselinecompuestoenrun,sinmodificarfuentes.750updatesadicionales,controllectorvsjointlector+encoder;headAdam/RNGheredados,encoderAdam nuevo1e-4,head.001,clip5porgruposeparado. Mismosbatches16original+16old7+16old9+16new9. Ambos1ciclo,gains/aristasfijos.

Recomputar TODAScachespara1ciclo;conjuntoonline4microbatch16 ygradientes,hacerparidadreal/gradienteacumulado antes. Último750fijo,holdoutsolo diagnóstico. Test5009+2507nuevoscompartidos,principal9joint-control,retención7secundaria. Unlector/un cerebro,no ciclosbiológicos. Agente servido intacto.

Runner experiments/train_early_interface.py,run runs/early-interface-001,reportero experiments/report_early_interface.py.
