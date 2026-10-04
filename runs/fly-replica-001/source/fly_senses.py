"""What the fly senses when it stands at a covered tile: the 5x5 neighbourhood as a bag of 'odorants'.

Each (relative position, tile state) pair is one odorant; states are wall, covered, or clue 0..8.
"""
import numpy as np

RADIUS=2;POSITIONS=[(dr,dc) for dr in range(-RADIUS,RADIUS+1) for dc in range(-RADIUS,RADIUS+1) if (dr,dc)!=(0,0)];STATES=11;ODORANTS=len(POSITIONS)*STATES

def windows(visible,size,silent=False):
    """visible: int8 (size*size,) with -1 covered, 0..8 clues. Returns covered cell ids and binary (cells, ODORANTS)."""
    grid=np.full((size+2*RADIUS,size+2*RADIUS),-2,np.int8);grid[RADIUS:-RADIUS,RADIUS:-RADIUS]=visible.reshape(size,size)
    cells=np.nonzero(visible==-1)[0];r,c=np.divmod(cells,size);out=np.zeros((len(cells),ODORANTS),np.float32)
    for p,(dr,dc) in enumerate(POSITIONS):
        state=grid[r+RADIUS+dr,c+RADIUS+dc].astype(np.int64)+2  # wall 0, covered 1, clues 2..10
        out[np.arange(len(cells)),p*STATES+state]=1
    if silent:out.reshape(len(cells),len(POSITIONS),STATES)[:,:,:2]=0  # only clue tiles smell; walls and covered tiles are odourless
    return cells,out

def plume(visible,size):
    """Arena physics instead of symbols: every open tile gives off a faint 'open' odour, and a clue tile N gives off
    N units of 'danger' odour that spread evenly over its still-covered neighbours. A fly standing on a covered tile
    smells two concentrations: total danger reaching it and how many open tiles surround it."""
    grid=np.full((size+2,size+2),-2,np.int16);grid[1:-1,1:-1]=visible.reshape(size,size);covered=(grid==-1).astype(np.float32);opened=(grid>=0).astype(np.float32)
    around=lambda a:sum(np.roll(np.roll(a,dr,0),dc,1) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0))
    share=np.where(opened>0,np.maximum(grid,0)/np.maximum(around(covered),1),0.).astype(np.float32)
    danger,count=around(share)[1:-1,1:-1].reshape(-1),around(opened)[1:-1,1:-1].reshape(-1);cells=np.nonzero(visible==-1)[0]
    return cells,np.stack([danger[cells],count[cells]],1).astype(np.float32)

SHARE_BINS=np.array([0,.13,.15,.18,.22,.27,.3,.35,.45,.55,.7,.9,1.01],np.float32)

def sniffs(visible,size,marked=None):
    """Sequential sniffing: standing on a covered tile the fly antennates each open neighbour in turn and smells that
    clue's danger share N/k on its own. Returns covered cells and, per cell, the shares of its open neighbours (nan = none)."""
    grid=np.full((size+2,size+2),-2,np.int16);grid[1:-1,1:-1]=visible.reshape(size,size);covered=(grid==-1).astype(np.float32)
    shifts=[(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)];around=lambda a:sum(np.roll(np.roll(a,dr,0),dc,1) for dr,dc in shifts)
    # A tile the fly has stress-marked soaks up one unit of each neighbouring clue's danger odour and stops receiving any.
    marks=np.zeros_like(covered)
    if marked is not None:marks[1:-1,1:-1]=marked.reshape(size,size)
    free=around(covered-marks);share=np.where((grid>=0)&(free>0),np.clip(grid-around(marks),0,None)/np.maximum(free,1),np.nan).astype(np.float32)
    stack=np.stack([np.roll(np.roll(share,dr,0),dc,1)[1:-1,1:-1].reshape(-1) for dr,dc in shifts],1);cells=np.nonzero(visible==-1)[0]
    return cells,stack[cells]


_SHIFTS=[(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)]

def quantise(count):
    """Weber-like resolution for large amounts: geometric steps of 25%."""
    return 0 if count<=0 else int(round(1.25**round(np.log(count)/np.log(1.25))))

def sniffs_far(visible,size,marked=None,mines=0):
    """Raw near sniffs plus a faint far field. Only on a tile with NO open neighbour (nothing strong to mask it) the fly smells
    three raw arena-wide amounts: M = the mine counter as displayed (never reduced by us), C = how much covered odour there is
    in total (coarse), P = how much of its own pheromone there is in total. No division, no subtraction.
    Returns (cells, 9, 4): eight near sniffs (N,k,m,0) and one far slot (M,C,P,1); nan where absent."""
    cells,near=sniffs_raw(visible,size,marked);out=np.full((len(cells),9,4),np.nan,np.float32);out[:,:8,:3]=near;out[:,:8,3]=np.where(np.isnan(near[:,:,0]),np.nan,0.)
    alone=np.isnan(near[:,:,0]).all(1);out[alone,8]=[mines,quantise(int((visible==-1).sum())),0 if marked is None else int(marked.sum()),1];return cells,out
sniffs_far.wants_counter=True

def sniffs_raw(visible,size,marked=None):
    """Raw senses, generic physics only. Every tile emits according to its OWN visible state: a clue emits its number N,
    a covered tile emits 'covered' odour, a tile the fly stress-marked also emits the mark pheromone. Odours add up over the
    eight surrounding tiles, the same in every direction. Antennating an open neighbour the fly smells three raw amounts:
    N of that clue, k = covered odour around it, m = own pheromone around it. No division, no subtraction, no rule knowledge.
    Returns covered cells and (cells, 8, 3) with nan where there is no open neighbour."""
    n=size;grid=np.full((n+4,n+4),-2,np.int16);grid[2:-2,2:-2]=visible.reshape(n,n);covered=(grid==-1).astype(np.float32);marks=np.zeros_like(covered)
    if marked is not None:marks[2:-2,2:-2]=marked.reshape(n,n)
    # slices of a doubly padded board instead of np.roll: same sums, no wrap-around, several times faster
    around=lambda a:sum(a[1+dr:n+3+dr,1+dc:n+3+dc] for dr,dc in _SHIFTS)            # (n+2,n+2): board plus a one-tile rim
    opened=grid[1:-1,1:-1]>=0;fields=[np.where(opened,v,np.nan).astype(np.float32) for v in (np.maximum(grid[1:-1,1:-1],0),around(covered),around(marks))]
    # neighbour in direction (dr,dc) as np.roll(np.roll(f,dr,0),dc,1) read it: the value at (r-dr, c-dc)
    stack=np.stack([np.stack([f[1-dr:n+1-dr,1-dc:n+1-dc].reshape(-1) for dr,dc in _SHIFTS],1) for f in fields],2);cells=np.nonzero(visible==-1)[0]
    return cells,stack[cells]


_DIRS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_SYM=[lambda r,c:(r,c),lambda r,c:(c,-r),lambda r,c:(-r,-c),lambda r,c:(-c,r),lambda r,c:(r,-c),lambda r,c:(-r,c),lambda r,c:(c,r),lambda r,c:(-c,-r)]
_GEOMETRY={}
for _d in _DIRS:
    for _ar in range(-2,3):
        for _ac in range(-2,3):
            if (_ar,_ac)!=(0,0):_GEOMETRY[_d,(_ar,_ac)]=min((f(-_d[0],-_d[1]),f(_ar,_ac)) for f in _SYM)
_CLASSES={g:i for i,g in enumerate(sorted(set(_GEOMETRY.values())))};GEOMETRIES=len(_CLASSES)

class PairStimulus:
    """Raw single sniffs plus a short working memory: for every open neighbour B of a tile, every other clue A within two tiles
    of B, as (N,k,m of B, N,k,m of A, how the fly had to move: tile->B and B->A, up to rotation and reflection). Nothing is resolved
    for the fly: whether two clues share unknown tiles, and what follows from it, has to be learned from heat and sugar."""
    def __init__(self,raw,owner,pairs):self.raw,self.owner,self.pairs=raw,owner,pairs
    def __getitem__(self,k):return self.raw[k]
    def __len__(self):return len(self.raw)

def sniffs_pairs(visible,size,marked=None):
    cells,raw=sniffs_raw(visible,size,marked);n=size;vis=visible.reshape(n,n);tri={}
    pad=np.full((n+2,n+2),-2,np.int16);pad[1:-1,1:-1]=vis;cov=(pad==-1).astype(np.int16);mk=np.zeros_like(cov)
    if marked is not None:mk[1:-1,1:-1]=marked.reshape(n,n)
    ksum=sum(cov[1+dr:n+1+dr,1+dc:n+1+dc] for dr,dc in _DIRS);msum=sum(mk[1+dr:n+1+dr,1+dc:n+1+dc] for dr,dc in _DIRS)
    owner=[];pairs=[]
    for i,cell in enumerate(cells.tolist()):
        if np.isnan(raw[i,:,0]).all():continue
        r,c=divmod(cell,n)
        for d in _DIRS:
            rr,cc=r+d[0],c+d[1]
            if not (0<=rr<n and 0<=cc<n) or vis[rr,cc]<=0:continue
            for ar in range(-2,3):
                a_r=rr+ar
                if not 0<=a_r<n:continue
                for ac in range(-2,3):
                    a_c=cc+ac
                    if (ar or ac) and 0<=a_c<n and vis[a_r,a_c]>0:
                        owner.append(i);pairs.append((vis[rr,cc],ksum[rr,cc],msum[rr,cc],vis[a_r,a_c],ksum[a_r,a_c],msum[a_r,a_c],_CLASSES[_GEOMETRY[d,(ar,ac)]]))
    return cells,PairStimulus(raw,np.array(owner,np.int64),np.array(pairs,np.int64).reshape(-1,7))
