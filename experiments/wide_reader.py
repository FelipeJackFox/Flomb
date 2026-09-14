"""Same-parameter reader: second convolution dilation1 versus3 (RF5 versus9)."""
from experiments.spatial_decoder import ActivityHead

def make_head(dilation=1):
 if dilation not in (1,3):raise ValueError(dilation)
 head=ActivityHead(True);head.net[2].dilation=(dilation,dilation);head.net[2].padding=(dilation,dilation)
 return head
