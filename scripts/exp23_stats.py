#!/usr/bin/env python3
"""exp23 - one table for every exp22 model, test set and threshold: mean ETS and frequency
bias with 95% bootstrap CIs (resampling seeds, then events within seeds), median volume
ratio, and the away-minus-home ETS difference with its CI. Writes reports/exp23_stats.md."""
import os, re, glob, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results'); B = 2000; rng = np.random.default_rng(0)
runs = {}
for f in glob.glob(os.path.join(R, 'result_exp22_*.json')):
    m = re.search(r'exp22_(\w+?)_(log|linear|global)_s(\d+)\.json', f)
    runs.setdefault((m.group(1), m.group(2)), []).append(json.load(open(f)))


def boot(per_seed):
    per_seed = [np.array([x for x in s if np.isfinite(x)]) for s in per_seed]
    per_seed = [s for s in per_seed if len(s)]
    if not per_seed: return float('nan'), float('nan'), float('nan'), []
    stats = []
    for _ in range(B):
        seeds = [per_seed[i] for i in rng.integers(0, len(per_seed), len(per_seed))]
        stats.append(np.mean(np.concatenate([s[rng.integers(0, len(s), len(s))] for s in seeds])))
    return float(np.mean(np.concatenate(per_seed))), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), stats


lines, summary = ['# exp23 - all models, all thresholds, bootstrap 95% CIs', ''], {}
encs = sorted({e for _, e in runs})
for enc in encs:
    for tr in ('point', 'distributed', 'inflow', 'breach'):
        rs = runs.get((tr, enc))
        if not rs: continue
        tests = list(rs[0]['scores'])
        thr_keys = [k for k in rs[0]['scores'][tr] if k != 'volume']
        lines += [f'## trained on {tr}, encoding {enc}, {len(rs)} seeds', '',
                  '| threshold | tested on | ETS mean [95% CI] | freq. bias mean [95% CI] | wet share | away - home ETS [95% CI] |', '|---|---|---|---|---|---|']
        for k in thr_keys:
            home = boot([[e['ets'] for e in r['scores'][tr][k]] for r in rs])
            for te in tests:
                if k not in rs[0]['scores'][te]: continue
                e = boot([[x['ets'] for x in r['scores'][te][k]] for r in rs])
                fb = boot([[x['fb'] for x in r['scores'][te][k]] for r in rs])
                wet = np.nanmedian([x['wet'] for r in rs for x in r['scores'][te][k]])
                diff = '-' if te == tr or not home[3] or not e[3] else \
                    f"{e[0]-home[0]:+.3f} [{np.percentile(np.array(e[3])-np.array(home[3]),2.5):+.3f}, {np.percentile(np.array(e[3])-np.array(home[3]),97.5):+.3f}]"
                lines.append(f"| {k} | {te}{' (own)' if te == tr else ''} | {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] | "
                             f"{fb[0]:.2f} [{fb[1]:.2f}, {fb[2]:.2f}] | {wet:.2f} | {diff} |")
                summary[f'{enc}|{tr}->{te}|{k}'] = {'ets': e[:3], 'fb': fb[:3], 'wet': float(wet)}
        vol = {te: float(np.nanmedian([v for r in rs for v in r['scores'][te]['volume']])) for te in tests}
        lines += ['', f"Median volume ratio: {', '.join(f'{te} {v:.2f}x' for te, v in vol.items())}", '']
        for te, v in vol.items(): summary[f'{enc}|{tr}->{te}|volume'] = v
open(os.path.join(ROOT, 'reports', 'exp23_stats.md'), 'w').write('\n'.join(lines))
json.dump(summary, open(os.path.join(R, 'result_exp23_stats.json'), 'w'), indent=1)
key = [l for l in lines if l.startswith('## ') or '| rel10 |' in l or '| phys0.1 |' in l or l.startswith('Median volume')]
print('\n'.join(key)); print('\nfull tables: reports/exp23_stats.md')
