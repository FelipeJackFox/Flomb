"""Run matched bounded pilots sequentially to avoid CPU/RAM contention."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from train_curriculum import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--episodes',type=int,default=64);p.add_argument('--eval-per-stratum',type=int,default=8);p.add_argument('--resume',action='store_true');a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=True);reports={}
 for mode in ['dagger','qrdqn','hybrid']:
  run=a.root/mode;command=[sys.executable,'-m','experiments.train','--mode',mode,'--run',str(run),'--episodes',str(a.episodes),'--eval-per-stratum',str(a.eval_per_stratum)]
  if (run/'config.json').exists():
   if not a.resume:raise RuntimeError('Existing suite requires --resume')
   if (run/'completed.json').exists():
    reports[mode]=json.loads((run/'evaluation-final.json').read_text());continue
   command+=['--resume']
  env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
  atomic_json(a.root/'suite-state.json',{'active':mode,'completed':list(reports),'started_at':time.time()})
  with (a.root/(mode+'.log')).open('a') as log:
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
   atomic_json(a.root/'suite-process.json',{'pid':os.getpid(),'child_pid':process.pid,'command':command})
   code=process.wait()
  if code or not (run/'completed.json').exists():raise RuntimeError(f'{mode} failed: exit {code}; see log')
  progress=json.loads((run/'progress.json').read_text())
  if progress['updates']==0:raise RuntimeError(f'{mode} completed without updates; increase pilot budget')
  reports[mode]=json.loads((run/'evaluation-final.json').read_text())
  print(mode,'complete',progress['updates'],'updates',flush=True)
 atomic_json(a.root/'comparison.json',{'evaluation':reports,'warning':'Small pilot; teacher-free greedy evaluation. Original policy requires matched greedy reevaluation before algorithm comparisons.'})
 atomic_json(a.root/'suite-state.json',{'active':None,'completed':list(reports),'finished_at':time.time()})
if __name__=='__main__':main()
