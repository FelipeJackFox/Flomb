"""Collector must execute learner actions, including mistakes, without teacher rescue."""
import contextlib,io,unittest
import numpy as np
import torch
from experiments.spatial_dagger import collect
from experiments.train import visible,teacher
from experiments.scaled_data import identity
from minesweeper import Minesweeper
from solver import analyze
class DummyMemo:
 def get(self,x,c):return torch.zeros(len(x),10,16,16)
class FirstLegal(torch.nn.Module):
 def forward(self,a,c):return -torch.arange(256,dtype=torch.float32)[None].expand(len(a),-1)
class CollectorTest(unittest.TestCase):
 def test_matches_independent_learner_trajectory(self):
  games=[dict(seed=s,size=7,mines=7,layout_hash=identity(Minesweeper(s,7,7))) for s in range(20)]
  with contextlib.redirect_stdout(io.StringIO()):rows,cache,stats=collect(DummyMemo(),FirstLegal(),games)
  expected={};clicks=wins=0
  for g in games:
   e=Minesweeper(g['seed'],7,7)
   while not e.done:
    native=int(np.flatnonzero(e.legal_mask())[0]);action=(native//7)*16+native%7
    info=analyze(e.observation(),e.legal_mask(),7,7)
    if info.safe:expected[(g['seed'],visible(e).tobytes())]=(action,teacher(e,info)[0])
    e.step(native);clicks+=1
   wins+=int(e.won)
  self.assertEqual(len(rows),len(expected));self.assertEqual(stats['clicks'],clicks);self.assertEqual(stats['wins'],wins)
  self.assertEqual(stats['teacher_actions'],0)
  self.assertTrue(any(not r['learner_safe'] for r in rows))
  for r in rows:
   action,labels=expected[(r['seed'],r['board'].tobytes())]
   self.assertEqual(r['learner_action'],action);np.testing.assert_array_equal(r['labels'],labels)
  self.assertEqual(len(cache[0]),len(rows))
if __name__=='__main__':unittest.main()
