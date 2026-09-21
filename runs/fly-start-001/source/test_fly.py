import unittest
import numpy as np
from experiments.fly_senses import sniffs
from experiments.mushroom_body import MushroomBody

class Tests(unittest.TestCase):
    def board(self):
        v=np.full(25,-1,np.int8);v[[12,11,13]]=[2,1,0];return v   # row 2: clue1 at col1, clue2 at col2, clue0 at col3
    def test_share_is_clue_over_covered_neighbours(self):
        cells,sh=sniffs(self.board(),5);row=sh[list(cells).index(6)]   # tile (1,1): touches clue1 (8 covered... ) and clue2
        v=self.board().reshape(5,5);k=lambda r,c:sum(v[a,b]==-1 for a in range(r-1,r+2) for b in range(c-1,c+2) if (a,b)!=(r,c))
        np.testing.assert_allclose(sorted(row[~np.isnan(row)].tolist()),sorted([1/k(2,1),2/k(2,2)]),rtol=1e-5)
    def test_far_tile_smells_nothing_and_padding_never_leaks(self):
        cells,sh=sniffs(self.board(),5);self.assertTrue(np.isnan(sh[list(cells).index(0)]).all());self.assertTrue(np.isnan(sh[list(cells).index(24)]).all())
    def test_mark_absorbs_one_unit_and_stops_receiving(self):
        marked=np.zeros(25,bool);marked[6]=True;cells,sh=sniffs(self.board(),5,marked);row=sh[list(cells).index(7)]   # (1,2) touches clue1? no: touches clues 1,2,0
        v=self.board().reshape(5,5);free=lambda r,c:sum(v[a,b]==-1 and not marked[a*5+b] for a in range(r-1,r+2) for b in range(c-1,c+2) if (a,b)!=(r,c))
        np.testing.assert_allclose(sorted(row[~np.isnan(row)].tolist()),sorted([(1-1)/free(2,1),(2-1)/free(2,2),0.]),rtol=1e-5)
    def test_raw_sniff_is_clue_covered_pheromone_without_arithmetic(self):
        from experiments.fly_senses import sniffs_raw
        marked=np.zeros(25,bool);marked[6]=True;cells,raw=sniffs_raw(self.board(),5,marked);row=raw[list(cells).index(7)];row=row[~np.isnan(row[:,0])]
        v=self.board().reshape(5,5);k=lambda r,c:sum(v[a,b]==-1 for a in range(r-1,r+2) for b in range(c-1,c+2) if (a,b)!=(r,c))
        m=lambda r,c:sum(marked[a*5+b] for a in range(r-1,r+2) for b in range(c-1,c+2) if (a,b)!=(r,c))
        self.assertEqual(sorted(map(tuple,row.astype(int).tolist())),sorted([(1,k(2,1),m(2,1)),(2,k(2,2),m(2,2)),(0,k(2,3),m(2,3))]))
        _,unmarked=sniffs_raw(self.board(),5);np.testing.assert_array_equal(np.nan_to_num(unmarked[...,:2]),np.nan_to_num(raw[...,:2]))   # marking never alters N or k
    def test_non_lethal_training_survives_burns_and_lethal_does_not(self):
        from experiments.fly_agents import play,TableAgent
        from experiments.fly_senses import sniffs_raw
        rng=np.random.default_rng(0);soft=[play(TableAgent(),s,7,7,rng,True,sense=sniffs_raw,lethal=False) for s in range(40)];hard=[play(TableAgent(),s,7,7,rng,True,sense=sniffs_raw) for s in range(40)]
        self.assertTrue(all(r['won'] for r in soft));self.assertGreater(sum(r['burns'] for r in soft),0);self.assertEqual(sum(r['burns'] for r in hard),0);self.assertLess(sum(r['won'] for r in hard),40)
    def test_far_field_is_raw_and_only_on_odourless_tiles(self):
        from experiments.fly_senses import sniffs_far,sniffs_raw,quantise
        marked=np.zeros(25,bool);marked[0]=True;cells,out=sniffs_far(self.board(),5,marked,mines=4);_,near=sniffs_raw(self.board(),5,marked)
        np.testing.assert_array_equal(np.nan_to_num(out[:,:8,:3],nan=-9),np.nan_to_num(near,nan=-9))
        far=out[list(cells).index(24),8];self.assertEqual(far.tolist(),[4,quantise(22),1,1]);self.assertTrue(np.isnan(out[list(cells).index(6),8]).all())
    def test_learned_marker_gate_and_credit(self):
        from experiments.fly_agents import LearnedMarker
        m=LearnedMarker(theta=-3.);z=np.array([-6.,-3.5,0.,5.]);self.assertEqual(m.decide(z,None,False).tolist(),[True,True,False,False])
        rng=np.random.default_rng(0);m.begin();act=m.decide(np.array([-3.]),rng,True);before=m.theta;m.reward(False)   # sugar after the decision
        self.assertEqual(m.theta>before,bool(act[0]));self.assertLess(abs(m.trace),abs((1-.5)/1.)+1e-9)
    def test_default_play_is_unchanged_by_marker_code(self):
        from experiments.fly_agents import play,TableAgent
        from experiments.fly_senses import sniffs_raw
        a=[play(TableAgent(),s,7,7,np.random.default_rng(1),True,sense=sniffs_raw) for s in range(30)];b=[play(TableAgent(),s,7,7,np.random.default_rng(1),True,sense=sniffs_raw,marker=None) for s in range(30)];self.assertEqual(a,b)
    def test_zero_start_opens_a_cascade_and_default_is_unchanged(self):
        from minesweeper import Minesweeper
        for seed in range(200):
            env=Minesweeper(seed,9,12,zero_start=True);self.assertEqual(int(env._mines.sum()),12);self.assertEqual(int(env.visible[40]),0);self.assertGreaterEqual(int((env.visible>=0).sum()),9)
            old=Minesweeper(seed,9,12);self.assertTrue(np.array_equal(old._mines,Minesweeper(seed,9,12,zero_start=False)._mines))
    def test_senses_never_read_mines(self):
        import inspect,experiments.fly_senses as m;self.assertNotIn('_mines',inspect.getsource(m))
    def test_paired_control_shares_odours_and_only_rewires(self):
        real,ctrl,other=MushroomBody(seed=3,tuning=.35),MushroomBody(seed=3,tuning=.35,shuffled=True),MushroomBody(seed=3,tuning=.35,shuffled=True,wiring_seed=9)
        for name in ('signature','tuned_mu','tuned_odour','plume_k'):np.testing.assert_array_equal(getattr(real,name),getattr(ctrl,name))
        self.assertFalse(np.array_equal(real.kc_mbon,ctrl.kc_mbon));self.assertFalse(np.array_equal(ctrl.kc_mbon,other.kc_mbon))
        np.testing.assert_allclose(np.sort(real.kc_mbon,1),np.sort(ctrl.kc_mbon,1));np.testing.assert_array_equal(real.sign,ctrl.sign)
    def test_context_reference_freezes_tonic_term_on_odorous_tiles(self):
        mb=MushroomBody(rule='rpe',tuning=.35,context_reference=True);h=mb.kenyon(np.array([[.5,1.]],np.float32))[0];mb.learn(h,True,0.);self.assertEqual(mb.baseline,0.)
        mb.learn_context(False);self.assertGreater(mb.baseline,0.)

if __name__=='__main__':unittest.main()
