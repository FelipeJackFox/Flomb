import unittest
import numpy as np
import torch
from experiments.backbone import encode_visible
from experiments.compare_input_representation import raw_map, RawPolicy
from experiments.spatial_decoder import ActivityHead

class InputRepresentationTest(unittest.TestCase):
    def test_public_clues_and_padding_keep_spatial_coordinates(self):
        board = np.full((16, 16), -2, dtype=np.int8)
        board[:7, :7] = -1
        board[0, 0], board[3, 5], board[6, 6] = 0, 4, 8
        x, legal = encode_visible([board.flatten()])
        x = torch.from_numpy(x); a = raw_map(x)
        self.assertEqual(a[0, 0, 2, 5].item(), 1)
        self.assertEqual(a[0, 1, 0, 0].item(), 1)
        self.assertEqual(a[0, 5, 3, 5].item(), 1)
        self.assertEqual(a[0, 9, 6, 6].item(), 1)
        self.assertEqual(a[0, :, 7:, :].count_nonzero().item(), 0)
        self.assertEqual(a[0, :, :, 7:].count_nonzero().item(), 0)
        np.testing.assert_array_equal(legal.reshape(16,16), board == -1)
        h = ActivityHead(True); c = torch.tensor([[7/16, 7/256]])
        torch.testing.assert_close(RawPolicy(h)(x,c), h(a,c), rtol=0, atol=0)

if __name__ == '__main__': unittest.main()
