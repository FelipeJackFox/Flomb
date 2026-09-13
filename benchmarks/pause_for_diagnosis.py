"""Bounded migration of the authorized hybrid run at its next saved checkpoint."""
import os,signal,json,pickle,time,shutil,subprocess
from pathlib import Path
r=Path('runs/hybrid-001');pid=json.loads((r/'live-process.json').read_text())['pid']
command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True)
assert '-m experiments.train' in command and '--run runs/hybrid-001' in command
initial=json.loads((r/'state.json').read_text())['episode'];deadline=time.monotonic()+100
while json.loads((r/'state.json').read_text())['episode']==initial:
 if time.monotonic()>deadline:raise TimeoutError('No new checkpoint; original trainer left running')
 time.sleep(.25)
os.kill(pid,signal.SIGSTOP)
try:
 with (r/'checkpoint.pkl').open('rb') as f:s=pickle.load(f)
 ep=s['episode'];backup=r/f'pre-diagnostic-{ep}';backup.mkdir(exist_ok=False)
 for name in ['checkpoint.pkl','state.json','progress.json','live-process.json','metrics.jsonl']:shutil.copy2(r/name,backup/name)
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
except BaseException:
 os.kill(pid,signal.SIGCONT);raise
# Keep abandoned post-checkpoint observations only in the preserved migration snapshot.
records=(r/'metrics.jsonl').read_text().splitlines();kept=[l for l in records if json.loads(l)['episode']<=ep]
tmp=r/'metrics-migration.tmp';tmp.write_text('\n'.join(kept)+'\n');tmp.replace(r/'metrics.jsonl')
if kept:(r/'progress.json').write_text(json.dumps(json.loads(kept[-1]),indent=2))

(r/'paused.json').write_text(json.dumps({'episode':ep,'reason':'User-approved learning diagnosis; do not auto-restart','time':time.time()},indent=2))
print(json.dumps({'paused_at':ep,'backup':str(backup)}))
