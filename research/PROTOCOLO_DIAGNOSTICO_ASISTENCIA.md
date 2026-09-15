# Deducciones omitidas frente a decisiones de riesgo

Adaptarencoder9 dio19/500vs17/500control (+.4pp,IC[-1,1.8]),sin mejora concluyente. Antes de más entrenamiento,medir cuánto rendimiento cambia al corregir errores de acción públicamente deducibles. Fijar latest-joint750 de nine-interface-001 como sujeto del diagnóstico,no como ganador validado.

500layouts NUEVOS9×9/12 compartidos,cuatro brazos sin entrenamiento:autónomo;corregir50%de propuestas que ignoran una segura certificada;corregir100%de esas propuestas;solver público en todas las acciones. Cuando se corrige se elige la segura de menor índice;en solver completo,si no hay segura se usa menor riesgo estimado delsolver,desempateíndice. El solver usa solo observación visible ycantidadtotaldeminas,sus probabilidades pueden ser heurísticas.

La moneda del50%es determinista por(seed,paso),independiente delcontenido oculto. Guardar cada estado visible,propuesta,acción e intervención para recalcular decisiones públicamente. Medir victorias,intervenciones/errores elegibles,accionesdelegadas ymuertesconseguradisponible. Comparaciones pareadas asistido−autónomo;una semilla/un cerebro y500layouts. Las trayectorias divergen,porloque NOson porcentajes causales aditivos de errores ni cotas superiores rigurosas.

Estos brazos asistidos no miden aprendizaje ni se sirven al usuario;son ablations del controlador para orientar el entrenamiento. No redefinir victorias por50/50 ni atribuir al cerebro la inteligencia delsolver. Preservar todos los checkpoints. Comprobarque brazoautónomo coincide con el evaluador existente enunlote antes de extenderloalos500;auditar decisiones del100%sinminasocultas.

Runner experiments/evaluate_assistance.py,run runs/assistance-diagnostic-001,reportero experiments/report_assistance.py.
