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
