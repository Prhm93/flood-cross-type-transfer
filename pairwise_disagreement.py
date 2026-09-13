import json, itertools, math

with open('results/result_exp6_metric_disagreement.json') as f:
    data = json.load(f)

def pairwise_disagreement(events):
    total_pairs, disagreements = 0, 0
    for a, b in itertools.combinations(events, 2):
        csi_a, csi_b = a['csi_final'], b['csi_final']
        if csi_a == csi_b:
            continue
        vol_a = abs(math.log10(max(a['volume_ratio'], 1e-6)))
        vol_b = abs(math.log10(max(b['volume_ratio'], 1e-6)))
        if vol_a == vol_b:
            continue
        csi_says_a_better = csi_a > csi_b
        vol_says_a_better = vol_a < vol_b
        total_pairs += 1
        if csi_says_a_better != vol_says_a_better:
            disagreements += 1
    pct = 100.0 * disagreements / total_pairs if total_pairs else float('nan')
    return disagreements, total_pairs, pct

print(f"{'Direction':<30s} {'Disagreements':>15s} {'Total pairs':>12s} {'Pct':>8s}")
for key, label in [('breach_home', 'Dam-break, own data'),
                    ('breach_away', 'Dam-break, rainfall data'),
                    ('harvey_home', 'Rainfall, own data'),
                    ('harvey_away', 'Rainfall, dam-break data')]:
    if key not in data:
        continue
    events = data[key]['events']
    d, t, pct = pairwise_disagreement(events)
    print(f"{label:<30s} {d:>15d} {t:>12d} {pct:>7.1f}%")
