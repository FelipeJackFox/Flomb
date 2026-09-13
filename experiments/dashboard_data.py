"""Read-only dashboard normalization; see DASHBOARD_DATA.md for its JSON contract."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import math
import threading

MODES = frozenset(('dagger', 'qrdqn', 'hybrid'))
_CACHE = {}
_LOCK = threading.RLock()


def _json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def registry(root=Path('runs')):
    """Explicit current runs, not arbitrary user paths or historical snapshots."""
    root = Path(root).resolve()
    found = {}
    if not root.is_dir():
        return found
    candidates = {root / 'curriculum-001'}
    # Direct run folders and one suite nesting level. Never traverse migrations.
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            candidates.add(child)
            for mode in MODES:
                candidates.add(child / mode)
    for path in sorted(candidates):
        if not path.is_dir() or path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        config = _json(path / 'config.json', {})
        if path == root / 'curriculum-001' or (isinstance(config, dict) and config.get('mode') in MODES):
            found[path.relative_to(root).as_posix()] = resolved
    return found


def resolve_run(root, run_id):
    """Resolve only a registered relative ID; no aliases, .., absolute paths."""
    if not isinstance(run_id, str) or not run_id or '\\' in run_id:
        raise ValueError('Invalid run ID')
    if Path(run_id).is_absolute() or any(p in ('', '.', '..') for p in run_id.split('/')):
        raise ValueError('Invalid run ID')
    runs = registry(root)
    if run_id not in runs:
        raise ValueError('Unknown run ID')
    return runs[run_id]


def _sig(paths):
    out = []
    for path in paths:
        try:
            s = path.stat()
            out.append((str(path), s.st_mtime_ns, s.st_size))
        except OSError:
            out.append((str(path), None, None))
    return tuple(out)


def _lines(path):
    try:
        with path.open() as src:
            for line in src:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                except ValueError:
                    # An in-progress final append must not break the dashboard.
                    continue
    except OSError:
        return


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _first(record, *keys):
    for key in keys:
        value = record.get(key)
        if _number(value):
            return value
    return None


def _ratio(a, b):
    return a / b if b else None


def _mean(records, key):
    values = [r[key] for r in records if _number(r.get(key))]
    return sum(values) / len(values) if values else None


# All other finite numeric game_metrics fields are additive counters. Ratios must
# be rebuilt from summed numerators/denominators, not averaged per episode.
GAME_RATIOS = {
    'safe_take_rate': ('safe_taken', 'safe_opportunities'),
    'exact_coverage': ('exact_moves', 'analyzed_moves'),
    'mean_excess_risk': ('excess_risk_sum', 'exact_moves'),
    'mean_chosen_risk': ('chosen_risk_sum', 'exact_moves'),
    'mean_minimum_risk': ('min_risk_sum', 'exact_moves'),
    'minimum_risk_choice_rate': ('minimum_risk_choices', 'exact_moves'),
    'logical_error_rate': ('logical_errors', 'analyzed_moves'),
    'revealed_per_click': ('revealed_cells', 'reveal_observations'),
    'flood_rate': ('flood_clicks', 'reveal_observations'),
    'frontier_rate': ('frontier_clicks', 'analyzed_moves'),
    'interior_rate': ('interior_clicks', 'analyzed_moves'),
    'border_rate': ('border_clicks', 'analyzed_moves'),
    'guess_survival_rate': ('exact_guess_survivals', 'exact_guess_observations'),
    'guess_expected_survival_rate': ('exact_guess_expected_survivals', 'exact_guess_observations'),
    'final_5050_neutral_score': ('final_5050_neutral_score_sum', 'final_5050_neutral_score_count'),
}


def _game(records):
    totals = {}
    excluded = set(GAME_RATIOS) | {'guess_survival_residual', 'schema_version'}
    count = 0
    for record in records:
        gm = record.get('game_metrics')
        if isinstance(gm, dict) and record.get('metrics_measured') is not False:
            count += 1
            for key, value in gm.items():
                if key not in excluded and _number(value):
                    totals[key] = totals.get(key, 0) + value
    if not count:
        return None
    ratios = {}
    for name, (numerator, denominator) in GAME_RATIOS.items():
        a, b = totals.get(numerator), totals.get(denominator)
        ratios[name] = {'numerator': a, 'denominator': b,
                        'value': _ratio(a, b) if a is not None and b is not None else None}
    actual, expected = totals.get('exact_guess_survivals'), totals.get('exact_guess_expected_survivals')
    denominator = totals.get('exact_guess_observations')
    numerator = actual - expected if actual is not None and expected is not None else None
    ratios['guess_survival_residual'] = {'numerator': numerator, 'denominator': denominator,
        'value': _ratio(numerator, denominator) if numerator is not None and denominator is not None else None}
    return {'measured_episodes': count, 'counters': totals, 'ratios': ratios}


def _normalize(record):
    new = dict(record)
    new['episode_steps'] = _first(record, 'episode_steps')
    if new['episode_steps'] is None and ('total_clicks' in record or 'training_wall_seconds' in record):
        new['episode_steps'] = _first(record, 'steps')
    new['steps'] = _first(record, 'total_clicks')
    if new['steps'] is None and 'episode_steps' in record:
        new['steps'] = _first(record, 'steps')
    new['training_seconds'] = _first(record, 'training_seconds', 'training_wall_seconds')
    new['rss'] = _first(record, 'rss', 'rss_bytes', 'peak_rss_bytes_macos')
    new['safe_fraction'] = _first(record, 'safe_fraction', 'mean_safe_fraction')
    new['return'] = _first(record, 'return', 'episode_return', 'reward_sum')
    losses = record.get('loss', {})
    if not isinstance(losses, dict):
        losses = {}
    new['loss_qr'] = _first(record, 'loss_qr')
    if new['loss_qr'] is None:
        new['loss_qr'] = _first(losses, 'qr')
    new['loss_teacher'] = _first(record, 'loss_teacher')
    if new['loss_teacher'] is None:
        new['loss_teacher'] = _first(losses, 'teacher')
    timing = record.get('timing', {})
    new['timing_metrics_seconds'] = _first(record, 'timing_metrics_seconds', 'metrics_seconds')
    if new['timing_metrics_seconds'] is None and isinstance(timing, dict):
        new['timing_metrics_seconds'] = _first(timing, 'metrics_seconds')
    return new


def _point(records, size=None, mines=None, bucket_end=None):
    last = records[-1]
    automatic = sum(bool(r.get('automatic_win', False)) for r in records)
    wins = sum(bool(r.get('won', False)) for r in records)
    policy_wins = sum(bool(r.get('won', False)) and not r.get('automatic_win', False) for r in records)
    policy_episodes = len(records) - automatic
    # Only teacher-observed episodes contribute to this action-weighted ratio.
    observed = [r for r in records if _number(r.get('teacher_actions')) and _number(r.get('episode_steps'))]
    teacher_actions = sum(r['teacher_actions'] for r in observed)
    teacher_steps = sum(r['episode_steps'] for r in observed)
    p = {'episode': last['episode'], 'bucket_end': bucket_end, 'size': size, 'mines': mines,
         'episodes': len(records), 'policy_episodes': policy_episodes, 'wins': wins,
         'policy_wins': policy_wins, 'automatic_wins': automatic,
         'win_rate': _ratio(policy_wins, policy_episodes),
         'teacher_fraction': _ratio(teacher_actions, teacher_steps),
         'teacher_actions': teacher_actions if observed else None,
         'teacher_observed_steps': teacher_steps if observed else None,
         'game_metrics': _game(records),
         'controller_metrics': {name: _game([{'game_metrics': r.get('controller_metrics', {}).get(name), 'metrics_measured': r.get('metrics_measured')}
                                            for r in records])
                                for name in ('policy', 'teacher', 'random')}}
    for field in ('safe_fraction', 'return', 'episode_steps', 'loss_qr', 'loss_teacher', 'gradient_norm', 'timing_metrics_seconds', 'epsilon', 'teacher_beta'):
        p[field] = _mean(records, field)
    for field in ('training_seconds', 'steps', 'updates', 'rss'):
        p[field] = next((r[field] for r in reversed(records) if _number(r.get(field))), None)
    return p


def _metric_records(path, cache):
    """Append-only JSONL tail cache, retaining a partial last line for next poll."""
    stat = path.stat()
    key = str(path)
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    old = cache.get(key)
    if old and old['signature'] == signature:
        return old['records']
    append = old and old['signature'][0] == stat.st_ino and stat.st_size > old['signature'][2]
    records = old['records'] if append else {}
    offset = old['offset'] if append else 0
    with path.open('rb') as src:
        src.seek(offset)
        while True:
            before = src.tell()
            line = src.readline()
            if not line or not line.endswith(b'\n'):
                offset = before
                break
            offset = src.tell()
            try:
                record = json.loads(line)
                ep = record.get('episode') if isinstance(record, dict) else None
                if type(ep) is int and ep >= 0:
                    records[ep] = _normalize(record)
            except (ValueError, UnicodeError):
                continue
    cache[key] = {'signature': signature, 'records': records, 'offset': offset}
    return records


def _aggregate(files, file_cache=None):
    unique = {}
    file_cache = file_cache if file_cache is not None else {}
    # Old resumed tails are superseded by later files and later lines. Unchanged
    # historical files never parse again; active files parse appended bytes only.
    for path in sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.name)):
        unique.update(_metric_records(path, file_cache))
    records = [unique[k] for k in sorted(unique)]
    if not records:
        return {'series': {}, 'size_series': {}, 'overall_series': [], 'bucket_size': 1, 'metric_episodes': 0}
    maximum = records[-1]['episode']
    # Short pilots retain single-episode data; long runs target <=150 points per
    # stratum. Around 30k episodes this is the requested 200-episode window.
    bucket_size = max(1, math.ceil(maximum / 150))
    strata, sizes, overall = {}, {}, {}
    for r in records:
        bucket = max(0, r['episode'] - 1) // bucket_size
        overall.setdefault(bucket, []).append(r)
        size, mines = r.get('size'), r.get('mines')
        if type(size) is int and type(mines) is int:
            strata.setdefault((size, mines), {}).setdefault(bucket, []).append(r)
            sizes.setdefault(size, {}).setdefault(bucket, []).append(r)
    series = {f'{size}x{size}-{mines}': [_point(rows, size, mines, (bucket + 1) * bucket_size) for bucket, rows in sorted(buckets.items())]
              for (size, mines), buckets in sorted(strata.items())}
    size_series = {f'{size}x{size}': [_point(rows, size=size, bucket_end=(bucket + 1) * bucket_size)
                                    for bucket, rows in sorted(buckets.items())]
                   for size, buckets in sorted(sizes.items())}
    # Sum original episode counters inside each exact stratum before ratios.
    by_exact={}
    for row in records:
        if type(row.get('size')) is int and type(row.get('mines')) is int:
            by_exact.setdefault((row['size'],row['mines']),[]).append(row)
    smoothed={}
    for (size,mines),rows in by_exact.items():
        indices=sorted(set(range(0,len(rows),max(1,math.ceil(len(rows)/100))))|{len(rows)-1})
        smoothed[f'{size}x{size}-{mines}']=[_point(rows[max(0,i-199):i+1],size,mines,rows[i]['episode']) for i in indices]
    return {'smoothed_series':smoothed,'smoothing_window':200,'series': series, 'size_series': size_series, 'overall_series': [_point(rows, bucket_end=(bucket + 1) * bucket_size) for bucket, rows in sorted(overall.items())],
            'bucket_size': bucket_size, 'metric_episodes': len(records)}


def _eval_results(results):
    if not isinstance(results, dict):
        return results
    result = {}
    for key, value in results.items():
        if isinstance(value, dict) and 'episodes' in value and 'wins' in value:
            item = dict(value)
            automatic = value.get('automatic_wins', 0)
            item['policy_episodes'] = value['episodes'] - automatic
            item['policy_wins'] = value['wins'] - automatic
            item['win_rate'] = _ratio(item['policy_wins'], item['policy_episodes'])
            item['safe_fraction'] = value.get('safe_fraction', value.get('mean_safe_fraction'))
            result[key] = item
        elif isinstance(value, dict):
            result[key] = _eval_results(value)
        else:
            result[key] = value
    return result


def _evaluations(run, completed, progress):
    found = {}
    for row in _lines(run / 'evaluations.jsonl'):
        episode = row.get('episode')
        if type(episode) is int:
            found[episode] = {'episode': episode, 'kind': 'periodic', 'results': _eval_results(row.get('results', {}))}
    out = [found[k] for k in sorted(found)]
    for name, kind, episode in [('evaluation-before.json', 'before', 0),
                                ('evaluation-final.json', 'final', (completed or {}).get('episodes', (progress or {}).get('episode')))]:
        data = _json(run / name)
        if data is not None:
            results = data.get('results', data)
            # Older baselines nest independent evaluators; preserve provenance
            # instead of combining their episodes or pretending these are strata.
            if isinstance(results, dict) and results and all(isinstance(v, dict) and 'episodes' not in v for v in results.values()):
                for actor, actor_results in results.items():
                    out.append({'episode': episode, 'kind': kind + '/' + actor, 'results': _eval_results(actor_results)})
            else:
                out.append({'episode': episode, 'kind': kind, 'results': _eval_results(results)})
    return sorted(out, key=lambda x: (x['episode'] if x['episode'] is not None else math.inf, 2 if x['kind'].startswith('final') else 1 if x['kind'] == 'periodic' else 0, x['kind']))


def _run_data(run, run_id):
    files = list(run.glob('metrics*.jsonl'))
    files = [p for p in files if p.is_file() and not p.is_symlink()]
    metric_sig = _sig(sorted(files))
    other = [run / name for name in ('config.json','progress.json','state.json','completed.json',
        'paused.json','control.json','control-applied.json','capabilities.json','evaluations.jsonl',
        'evaluation-before.json','evaluation-final.json')]
    signature = _sig(other)
    cache = _CACHE.setdefault(str(run), {})
    if cache.get('metric_sig') != metric_sig:
        cache['metrics'] = _aggregate(files, cache.setdefault('metric_files', {}))
        cache['metric_sig'] = metric_sig
        cache.pop('data_sig', None)
    if cache.get('data_sig') != signature:
        data = {'id': run_id}
        for key in ('config','progress','state','completed','paused','control'):
            data[key] = _json(run / (key + '.json'))
        data['applied'] = _json(run / 'control-applied.json')
        capabilities = _json(run / 'capabilities.json', {})
        data['control_supported'] = ((data['config'] or {}).get('mode') in MODES
                                     and capabilities.get('runtime_curriculum') is True)
        data.update(cache['metrics'])
        data['evaluations'] = _evaluations(run, data['completed'], data['progress'])
        cache['data'] = data
        cache['data_sig'] = signature
    return cache['data']


def build_dashboard(root=Path('runs')):
    """Return JSON-compatible data; caller may serialize or modify its own copy."""
    with _LOCK:
        runs = [_run_data(path, run_id) for run_id, path in registry(root).items()]
        return {'runs': copy.deepcopy(runs),
                'generated_at': datetime.now(timezone.utc).isoformat()}
