import unittest
from experiments.select_by_games import choose
class TestSelection(unittest.TestCase):
 def test_wins_first_then_fewer_updates(self):
  c={'initial':{'step':0},'best':{'step':500},'last':{'step':1500}};keys=list(c)
  self.assertEqual(choose(keys,c,{'initial':{'wins':10},'best':{'wins':11},'last':{'wins':12}}),'last')
  self.assertEqual(choose(keys,c,{k:{'wins':10} for k in keys}),'initial')
  self.assertEqual(choose(keys,c,{'initial':{'wins':9},'best':{'wins':10},'last':{'wins':10}}),'best')
if __name__=='__main__':unittest.main()
