# Adaptar el codificador a9×9 conservando repaso7

La réplica de repaso7 mostró+3.6pp frentecontrol en7(IC[.6,6.6]) y diferencia9+.4pp(IC[-.3,1.1]);retención prometedora,pero éxito9siguebajo. El codificador histórico se adaptó con5/7 y permaneció congelado durante toda la experiencia9. Hipótesis: adaptarlo a los ejemplos9 aporta más que seguir entrenando soloel lector.

Piloto02desde `runs/seven-rehearsal-001/latest-replay.pt`. Dosbrazos750updates adicionales:controlsololector;conjuntolector+encoder. Mismos pesos/Adamlector/RNG/índices,mezcla16originales+16DAgger7+32experiencia9fija. LectorAdam heredado(.001),encoderAdam nuevo(.0001);clip5porgrupo separado. Encoder/grafoinicialidéntico;grafo,gainsneuronal y aristas permanecen fijos en ambos.

Conjunto calcula actividad y gradiente online con4microbatches16porupdate;controlusa caches verificadas. No reutilizar cachesdelencoder antiguo para entrenar/evaluar el conjunto tras actualizarlo. Extender JointInterface a tamaños con tablas registradas,rechazarlos si faltan. Antesdeentrenar:paridadforwardconactivity_map enmezcla5/7/9,gradientes acumuladosvsbatchcompleto ygradienteencoder positivo enejemplos9.

Último750fijo para ambos,holdoutpequeño solo diagnóstico. Test5009+2507nuevos,principal9joint−control;secundarias7retención yvsbaselinepadreconrepaso. Una semilla/un cerebro,750layouts compartidos;sin ventaja anatómica ni éxito garantizado. Fuente/estados/hash/Adam/RNGpreservados. Runner experiments/train_nine_interface.py,run runs/nine-interface-001. Agente servido intacto.
