"""Read completed paired results; never change training or select checkpoints."""
import json,pickle,hashlib
from pathlib import Path
from scipy.stats import binomtest
r=Path('runs/spatial-diagnostic-001')
assert (r/'completed.json').exists()
results=json.loads((r/'results.json').read_text());manifest=json.loads((r/'manifest.json').read_text())
assert hashlib.sha256(Path('runs/hybrid-001/checkpoint.pkl').read_bytes()).hexdigest()==manifest['paused_hybrid_sha256']
with (r/'dataset.pkl').open('rb') as f:data=pickle.load(f)
known={(x['size'],x['board'].tobytes()) for x in data['train']}
novel=[i for i,row in enumerate(data['holdout']) if (row['size'],row['board'].tobytes()) not in known]
lines=['# Resultado de la comparación espacial','','Prueba exploratoria terminada. Ambas variantes entrenaron el cerebro completo200actualizaciones por semilla, con256posiciones y minibatches emparejados. Dos semillas. Codificador local3x3 fijo y sin parámetros adicionales; no es una CNN aprendida. Híbrido pausado en14,384: SHA256 confirmado sin cambios.','','| Semilla | Entrada | Conocidas /256 | Reservadas /128 | Observaciones nuevas /'+str(len(novel))+' | Victorias5x5 | Victorias7x7 |','|---|---|---:|---:|---:|---:|---:|']
for x in sorted(results,key=lambda x:(x['seed'],x['variant'])):
 scores=[]
 for size in [5,7]:
  games=[g for g in x['games'] if g['size']==size and not g['automatic']];scores.append(f"{sum(g['won'] for g in games)}/{len(games)}")
 lines.append(f"|{x['seed']}|{x['variant']}|{x['train']['correct']}|{x['holdout']['correct']}|{sum(x['holdout']['hits'][i] for i in novel)}|{scores[0]}|{scores[1]}|")
paired=[]
for seed in manifest['seeds']:
 a=next(x for x in results if x['seed']==seed and x['variant']=='raw');b=next(x for x in results if x['seed']==seed and x['variant']=='local')
 plus=sum(y and not x for x,y in zip(a['holdout']['hits'],b['holdout']['hits']));minus=sum(x and not y for x,y in zip(a['holdout']['hits'],b['holdout']['hits']))
 paired.append({'seed':seed,'local_only_correct':plus,'raw_only_correct':minus,'mcnemar_exact_p':float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.})
lines+=['','## Comparación emparejada','',*[f"- Semilla{x['seed']}: solo local acierta{x['local_only_correct']}; solo actual acierta{x['raw_only_correct']}; McNemar exacto bilateral p={x['mcnemar_exact_p']:.4f}." for x in paired],'','## Límites e interpretación','','Los splits son disjuntos por distribución real de minas. Dos posiciones reservadas tienen el mismo estado visible que alguna de entrenamiento pese a pertenecer a tableros distintos; se muestra también el análisis excluyéndolas. Las partidas completas usan un tercer conjunto disjunto. Ninguna recompensa, etiqueta ni entrada del modelo consulta ubicaciones ocultas. El hash de minas se utiliza exclusivamente al separar los datasets.','','No se deben sumar las dos semillas como256tableros independientes: se repiten los mismos128. Tampoco equivale acertar posiciones del solver a ganar trayectorias autónomas. Una ausencia de ventaja aquí solo se refiere a este promedio local fijo, presupuesto y datos; no descarta codificadores convolucionales aprendidos ni demuestra que la conectividad sea irrelevante. Ver PROTOCOLO_ENTRADA_ESPACIAL.md para detalles.']
Path('research/RESULTADO_ENTRADA_ESPACIAL.md').write_text('\n'.join(lines)+'\n');(r/'paired.json').write_text(json.dumps(paired,indent=2));print('\n'.join(lines[:13]));print(json.dumps(paired))
