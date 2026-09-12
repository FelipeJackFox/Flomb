"""Episode-boundary curriculum requests; no process or trainer control.

control.json is the requested state, control-applied.json the trainer's explicit
acknowledgement. Missing control preserves the original scheduled RNG stream.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import numpy as np
from curriculum import draw_board

STRATA = (('small', 5, 3), ('small', 7, 7), ('medium', 9, 10),
          ('medium', 12, 24), ('large', 16, 40), ('large', 16, 56))
DEFAULT_WEIGHTS = [1 / 6] * 6


def validate_control(payload):
    """Return a fresh normalized public payload; reject unknown keys/paths."""
    if not isinstance(payload, dict) or set(payload) != {'mode', 'weights'}:
        raise ValueError('Expected only mode and weights')
    if payload['mode'] not in ('scheduled', 'manual'):
        raise ValueError('mode must be scheduled or manual')
    values = payload['weights']
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError('weights must contain six numbers')
    if any(isinstance(x, bool) or not isinstance(x, Real)
           or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('weights must be finite nonnegative numbers')
    # Scale first, avoiding overflow for otherwise valid finite inputs.
    maximum = max(values)
    if maximum <= 0:
        raise ValueError('At least one weight must be positive')
    scaled = [float(x / maximum) for x in values]
    total = math.fsum(scaled)
    return {'mode': payload['mode'], 'weights': [x / total for x in scaled]}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(data, out, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _locked(run):
    run = Path(run)
    if not run.is_dir():
        raise ValueError('Run directory must already exist')
    with (run / '.curriculum-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield run
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(run):
    path = Path(run) / 'control.json'
    if not path.exists():
        return {'mode': 'scheduled', 'weights': DEFAULT_WEIGHTS.copy(), 'revision': 0}
    data = json.loads(path.read_text())
    normalized = validate_control({k: data[k] for k in ('mode', 'weights')})
    rev = data.get('revision')
    if type(rev) is not int or rev < 1:
        raise ValueError('Invalid control revision')
    return {**normalized, 'revision': rev}


def request_control(run_path, payload):
    """Persist a request, without claiming adoption or touching training state."""
    normalized = validate_control(payload)
    with _locked(run_path) as run:
        state = {**normalized, 'revision': _read(run)['revision'] + 1,
                 'requested_at': _now()}
        _atomic(run / 'control.json', state)
        with (run / 'control-audit.jsonl').open('a') as log:
            log.write(json.dumps({'event': 'requested', **state}, allow_nan=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
    return state


def sample_board(rng, episode, horizon, run_path):
    """Read requested state once per episode. Scheduled consumes no extra RNG."""
    control = _read(run_path)
    metadata = {**control, 'weights': control['weights'].copy()}
    if control['mode'] == 'scheduled':
        group, size, mines, probs = draw_board(rng, episode, horizon)
        return group, size, mines, probs, metadata
    index = int(rng.choice(len(STRATA), p=control['weights']))
    group, size, mines = STRATA[index]
    weights = control['weights']
    probs = np.array([sum(weights[:2]), sum(weights[2:4]), sum(weights[4:])])
    return group, size, mines, probs, metadata


def applied_control(run_path, metadata, episode):
    """Trainer calls this only after adopting sampled metadata for an episode.

    An older in-flight episode may acknowledge an earlier revision while a newer
    request is pending. Never mark the newer request adopted in its place.
    """
    normalized = validate_control({k: metadata[k] for k in ('mode', 'weights')})
    revision = metadata.get('revision')
    if type(revision) is not int or revision < 0:
        raise ValueError('Invalid applied revision')
    if type(episode) is not int or episode < 0:
        raise ValueError('episode must be a nonnegative integer')
    with _locked(run_path) as run:
        requested = _read(run)
        if revision > requested['revision']:
            raise ValueError('Cannot acknowledge an unrequested revision')
        path = run / 'control-applied.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['revision'] >= revision:
                return previous
        state = {**normalized, 'revision': revision, 'episode': episode,
                 'applied_at': _now()}
        _atomic(path, state)
    return state
