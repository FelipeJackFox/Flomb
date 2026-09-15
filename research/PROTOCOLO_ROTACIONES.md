# Simetría del tablero en inferencia

Modelo early-interface-001/latest-joint fijo,1ciclo,head yencoder originales. Comparar identidad vs promedio logits de4rotaciones0/90/180/270,deshaciendo rotación de salidas. Rotar solamente subtableroNxN;conservar padding ycontexto. Legalidad aplicada por evaluador después de combinar. Sin maestro/updates ni cambio pesos.

750layouts nuevos5009/2507 compartidos. Principal9rotado-identidad,secundario7,bootstrap pareado10000. Presupuesto1500episodios,brazo rotado4inferencias por decisión. Es diagnóstico/ensemble, no aprendizaje ni comparación de cómputo igualado. Validar ida/vuelta, padding,modelo equivarante ycovarianza ensemble antes de ejecutar. Fuentes/agente servido intactos.
