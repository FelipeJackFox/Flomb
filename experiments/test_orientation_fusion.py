import unittest
import torch
from experiments.orientation_fusion import rotate_sized,aligned_stack,FusionHead,FusionPolicy
from experiments.rotation_readout import rotate_grid,RotationPolicy

def boards(sizes):
    torch.manual_seed(3);x=torch.zeros(len(sizes),16,16,10);c=torch.zeros(len(sizes),2)
    for i,n in enumerate(sizes):x[i,:n,:n]=torch.rand(n,n,10);c[i,0]=n/16;c[i,1]=.1
    return x,c

class FakeMemo:
    """Deliberately non-equivariant 'brain': position-dependent gain on the clues."""
    def __init__(self):torch.manual_seed(5);self.gain=torch.rand(10,16,16)
    def get(self,x,c):return x.reshape(-1,16,16,10).permute(0,3,1,2)*self.gain

class Tests(unittest.TestCase):
    def test_rotation_matches_reference_and_keeps_padding(self):
        x,c=boards([5,7,9,16,7])
        for k in range(-3,4):
            ours=rotate_sized(x.permute(0,3,1,2),c,k).permute(0,2,3,1)
            torch.testing.assert_close(ours,rotate_grid(x,c,k),rtol=0,atol=0)
        self.assertEqual(float(rotate_sized(x.permute(0,3,1,2),c,1)[0,:,5:].abs().sum()),0.)
    def test_inverse(self):
        x,c=boards([7,9]);g=x.permute(0,3,1,2)
        torch.testing.assert_close(rotate_sized(rotate_sized(g,c,3),c,-3),g,rtol=0,atol=0)
    def views(self,memo,x,c):
        return torch.stack([memo.get(rotate_grid(x,c,m).reshape(-1,2560),c) for m in range(4)],1)
    def test_cached_turn_equals_recomputed_rotated_board(self):
        x,c=boards([7,9,5]);memo=FakeMemo();a4=self.views(memo,x,c)
        for j in range(4):
            direct=aligned_stack(self.views(memo,rotate_grid(x,c,j),c),c)
            torch.testing.assert_close(aligned_stack(a4,c,torch.full((3,),j)),direct,rtol=0,atol=0)
            rolled=rotate_sized(aligned_stack(a4,c).reshape(3,4,10,16,16).roll(-j,1).reshape(3,40,16,16),c,j)
            torch.testing.assert_close(direct,rolled,rtol=0,atol=0)
    def test_policy_matches_training_stack_and_parameters(self):
        x,c=boards([7,9]);memo=FakeMemo();head=FusionHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()),12739)
        expected=head(aligned_stack(self.views(memo,x,c),c),c)
        torch.testing.assert_close(FusionPolicy(memo,head)(x.reshape(-1,2560),c),expected,rtol=0,atol=0)
    def test_mean_of_logits_is_a_special_case_frame(self):
        # RotationPolicy undoes rotations on logits exactly as the stack does on activity.
        x,c=boards([7]);memo=FakeMemo()
        class Sum(torch.nn.Module):
            def forward(self,xx,cc):return memo.get(xx,cc).sum(1).flatten(1)
        mean=RotationPolicy(Sum())(x.reshape(-1,2560),c)
        stack=aligned_stack(self.views(memo,x,c),c)
        torch.testing.assert_close(mean,(stack.sum(1)/4).flatten(1),rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
