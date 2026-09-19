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
