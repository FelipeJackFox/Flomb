"""Visible-only, observational Minesweeper metrics; never accepts an environment.

Call analyze_move *before* stepping. Afterwards add ``revealed_delta`` (number of
new visible cells) and ``survived`` (bool). All summaries expose additive counters
and explicit denominators; combine counters across episodes, never average rates.
Solver enumeration is bounded at its normal 12 frontier cells. Unknown/heuristic
probabilities remain None, not zero. These diagnostics do not change rewards.
"""
import numpy as np
from solver import analyze

EXACT = 'exact_uniform_consistent_boards'


def analyze_move(visible, size, mine_count, action, analysis=None):
    """Analyze an unpadded flat/2-D integer board (-1 hidden, 0..8 clue).

    Action uses the native size*size index, not the 16x16 policy canvas. Only
    public clues and total mine count are accepted. Invalid inputs raise rather
    than quietly introduce biased measurements. This may be sampled for speed;
    unsampled actions must not be counted as solver failures or safe play.
    Optional ``analysis`` must be the solver Analysis for this exact preaction
    board/mine_count (not a stale state). Reuse the teacher analysis to avoid
    enumeration twice; matching legal keys are checked, clues cannot be rechecked
    without recomputing the analysis.
    """
    arr = np.asarray(visible)
    if arr.size != size * size or not np.all(np.isin(arr, range(-1, 9))):
        raise ValueError('Expected unpadded visible clues -1..8')
    arr = arr.astype(np.int8).reshape(-1)
    if not 0 < mine_count < size * size or not 0 <= action < arr.size or arr[action] != -1:
        raise ValueError('Invalid mine count or non-hidden action')
    legal = arr == -1
    if analysis is None:
        obs = np.eye(10, dtype=np.float32)[arr + 1]
        info = analyze(obs, legal, size, mine_count)
    else:
        info = analysis
        hidden = set(map(int, np.flatnonzero(legal)))
        if (set(info.mine_probability) != hidden or not info.safe <= hidden
                or not info.mines <= hidden or info.safe & info.mines):
            raise ValueError('Precomputed analysis does not match hidden cells')
    exact = info.probability_method == EXACT
    chosen = float(info.mine_probability[action]) if exact else None
    minimum = min(map(float, info.mine_probability.values())) if exact else None
    r, c = divmod(int(action), size)
    frontier = any(arr[rr * size + cc] >= 0
                   for rr in range(max(0, r - 1), min(size, r + 2))
                   for cc in range(max(0, c - 1), min(size, c + 2)))
    safe_opportunity = bool(info.safe)
    unresolved = set(map(int, np.flatnonzero(legal))) - info.mines
    final5050 = (exact and not info.safe and len(unresolved) == 2
                 and mine_count - len(info.mines) == 1
                 and int(action) in unresolved
                 and all(abs(info.mine_probability[i] - .5) < 1e-12 for i in unresolved))
    return dict(schema_version=1, size=int(size), mine_count=int(mine_count),
                action=int(action), hidden_count=int(legal.sum()),
                classification=info.classify(int(action)),
                probability_method=info.probability_method, exact_probability=exact,
                chosen_mine_risk=chosen, min_mine_risk=minimum,
                excess_risk=max(0., chosen - minimum) if exact else None,
                safe_opportunity=safe_opportunity, safe_taken=int(action) in info.safe,
                proven_mine_chosen=int(action) in info.mines,
                logical_error=(safe_opportunity and int(action) not in info.safe) or int(action) in info.mines,
                frontier=frontier, interior=not frontier,
                border=r in (0, size - 1) or c in (0, size - 1),
                final_5050_candidate=final5050)


def summarize_moves(moves, won, automatic_win=False):
    """Return additive counters, denominators and convenient per-episode ratios.

    `moves` must cover the entire episode to certify a final 50/50: set
    ``metrics_sampled=True`` in any record if records were sampled. A final 50/50
    contributes .5 on either outcome only with no detected earlier logical error.
    It is a variance-reduced *separate score*, not an adjusted reward or a win.
    Solver incompleteness means 'no detected error' does not certify optimal play.
    For partially sampled episodes pass a sentinel {metrics_sampled: True}.
    """
    records = [m for m in moves if 'action' in m]
    exact = [m for m in records if m.get('exact_probability')]
    opportunities = [m for m in records if m.get('safe_opportunity')]
    guesses = [m for m in exact if m.get('classification') == 'guess' and 'survived' in m]
    expansion = [m for m in records if 'revealed_delta' in m]
    errors = sum(bool(m.get('logical_error')) for m in records)
    complete = not any(m.get('metrics_sampled', False) for m in moves)
    certified = bool(records and complete and not automatic_win and not errors
                     and records[-1].get('final_5050_candidate')
                     and 'survived' in records[-1]
                     and bool(records[-1]['survived']) == bool(won))
    out = dict(episodes=1, wins=int(bool(won)), automatic_wins=int(bool(automatic_win)),
               policy_episodes=int(not automatic_win), policy_wins=int(bool(won) and not automatic_win),
               analyzed_moves=len(records), exact_moves=len(exact),
               safe_opportunities=len(opportunities),
               safe_taken=sum(bool(m.get('safe_taken')) for m in opportunities),
               proven_mine_clicks=sum(bool(m.get('proven_mine_chosen')) for m in records),
               logical_errors=errors,
               chosen_risk_sum=sum(m['chosen_mine_risk'] for m in exact),
               min_risk_sum=sum(m['min_mine_risk'] for m in exact),
               excess_risk_sum=sum(m['excess_risk'] for m in exact),
               minimum_risk_choices=sum(m['excess_risk'] <= 1e-12 for m in exact),
               frontier_clicks=sum(bool(m.get('frontier')) for m in records),
               interior_clicks=sum(bool(m.get('interior')) for m in records),
               border_clicks=sum(bool(m.get('border')) for m in records),
               reveal_observations=len(expansion),
               revealed_cells=sum(int(m['revealed_delta']) for m in expansion),
               flood_clicks=sum(int(m['revealed_delta']) > 1 for m in expansion),
               exact_guess_observations=len(guesses),
               exact_guess_survivals=sum(bool(m['survived']) for m in guesses),
               exact_guess_expected_survivals=sum(1 - m['chosen_mine_risk'] for m in guesses),
               exact_guess_variance_sum=sum(m['chosen_mine_risk'] * (1 - m['chosen_mine_risk']) for m in guesses),
               final_5050_certified=int(certified), final_5050_wins=int(certified and won),
               final_5050_neutral_score_sum=.5 if certified else float(bool(won)),
               final_5050_neutral_score_count=int(not automatic_win))
    if automatic_win:
        out['final_5050_neutral_score_sum'] = 0.
    def ratio(a, b):
        return out[a] / out[b] if out[b] else None
    out.update(safe_take_rate=ratio('safe_taken', 'safe_opportunities'),
               exact_coverage=ratio('exact_moves', 'analyzed_moves'),
               mean_excess_risk=ratio('excess_risk_sum', 'exact_moves'),
               revealed_per_click=ratio('revealed_cells', 'reveal_observations'),
               guess_survival_residual=(out['exact_guess_survivals'] - out['exact_guess_expected_survivals']) / len(guesses) if guesses else None)
    return out
