"""
exp_suite.py
============

Experiments 1, 2, 4 and 5, run from one file so you can launch and walk away.

  EXP 1  Threshold sweep      — is the CSI paradox real, or an artefact of
                                one threshold choice? (no training needed)
  EXP 2  Rainfall ablation    — does transfer still fail when NEITHER dataset
                                has a rainfall channel? (rules out "you just
                                removed an input")
  EXP 4  Fine-tuning curve    — how many INDEPENDENT target events are needed?
                                Compared against training from scratch.
  EXP 5  Stronger model       — does the transfer failure survive a bigger
                                network? (rules out "your model was too weak")

Run one:
    python3 scripts/exp_suite.py --only 1
Run all:
    python3 scripts/exp_suite.py
Smoke test:
    python3 scripts/exp_suite.py --quick
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_lib import (load_cache, build_cache, set_seed, train_model,
                                 build_model, get_device, rollout_predictions,
                                 score_rollouts, volume_ratio, evaluate,
                                 strip_rainfall, CACHE_PATH)

OUT = 'results'


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'result_{name}.json')
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {p}")


# ================================================================== #
# EXPERIMENT 1 — THRESHOLD SWEEP
# ================================================================== #

def exp1_threshold_sweep(data, device, args):
    """
    Re-score EXISTING trained models at many wet/dry thresholds.

    The reviewer's objection: 'you found the CSI paradox at one threshold
    (1% of max depth) — did you manufacture it?'

    Answer: score at 0.1%, 0.5%, 1%, 2%, 5%, 10% of max depth, plus fixed
    absolute thresholds. If the paradox holds across all of them, it is real.
    Mass/volume ratio is threshold-free and confirms independently.
    """
    print("\n" + "=" * 66)
    print("  EXPERIMENT 1 — CSI THRESHOLD SENSITIVITY")
    print("=" * 66)

    rel_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    fixed_thresholds = [0.01, 0.05]

    out = {'relative': {}, 'fixed': {}, 'volume_ratio': {}}

    for train_on in ['breach', 'harvey']:
        away = 'harvey' if train_on == 'breach' else 'breach'
        ckpt = os.path.join(OUT, f'best_{train_on}_vector_s0.pt')
        if not os.path.exists(ckpt):
            print(f"  [skip] no checkpoint {ckpt} — train first")
            continue

        model = build_model('vector', 64, 3, device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                          weights_only=True))

        print(f"\n  Model trained on {train_on}:")
        roll_home = rollout_predictions(model, data[train_on]['test'], device,
                                         args.rollout_steps, args.eval_samples)
        roll_away = rollout_predictions(model, data[away]['test'], device,
                                         args.rollout_steps, args.eval_samples)

        vr_home = volume_ratio(roll_home)
        vr_away = volume_ratio(roll_away)
        out['volume_ratio'][train_on] = {'home': vr_home, 'away': vr_away}

        print(f"    VOLUME RATIO (threshold-free): home {vr_home:.2f}x  "
              f"away {vr_away:.2f}x   (1.0 = perfect)")
        print(f"    {'threshold':>12s} {'home CSI':>10s} {'away CSI':>10s}  verdict")

        for thr in rel_thresholds:
            h = score_rollouts(roll_home, 'relative', thr)
            a = score_rollouts(roll_away, 'relative', thr)
            paradox = "AWAY>HOME (paradox)" if a['csi_final'] > h['csi_final'] else ""
            print(f"    {thr*100:>10.1f}% {h['csi_final']:>10.4f} "
                  f"{a['csi_final']:>10.4f}  {paradox}")
            out['relative'].setdefault(train_on, {})[str(thr)] = {
                'home_csi': h['csi_final'], 'away_csi': a['csi_final'],
                'home_mass': h['mass_error'], 'away_mass': a['mass_error']}

        for thr in fixed_thresholds:
            h = score_rollouts(roll_home, 'fixed', thr)
            a = score_rollouts(roll_away, 'fixed', thr)
            paradox = "AWAY>HOME (paradox)" if a['csi_final'] > h['csi_final'] else ""
            print(f"    fixed {thr:<5.2f} {h['csi_final']:>10.4f} "
                  f"{a['csi_final']:>10.4f}  {paradox}")
            out['fixed'].setdefault(train_on, {})[str(thr)] = {
                'home_csi': h['csi_final'], 'away_csi': a['csi_final']}

    print("\n  READ THIS: if 'paradox' appears at most thresholds for the")
    print("  breach-trained model, the CSI failure is NOT a threshold artefact.")
    print("  The volume ratio confirms it without any threshold at all.")
    save('exp1_threshold_sweep', out)
    return out


# ================================================================== #
# EXPERIMENT 2 — RAINFALL ABLATION
# ================================================================== #

def exp2_rainfall_ablation(data, device, args):
    """
    Retrain with the rainfall channel zeroed in BOTH datasets.

    The reviewer's objection: 'a rainfall-trained model fails on breach data
    simply because its rainfall input vanished — that is not flood-type
    transfer failure.'

    Answer: remove rainfall from both. If transfer still fails, the failure
    is not about the missing input channel.
    """
    print("\n" + "=" * 66)
    print("  EXPERIMENT 2 — RAINFALL ABLATION (no rain in either dataset)")
    print("=" * 66)

    nr = {k: {s: strip_rainfall(v) for s, v in d.items()}
          for k, d in data.items() if k in ('breach', 'harvey')}

    results = []
    for train_on in ['breach', 'harvey']:
        away = 'harvey' if train_on == 'breach' else 'breach'
        for seed in args.seeds:
            print(f"\n  --- no-rain, train on {train_on}, seed {seed} ---")
            set_seed(seed)
            m = build_model('vector', 64, 3, device)
            ck = os.path.join(OUT, f'norain_{train_on}_s{seed}.pt')
            train_model(m, nr[train_on]['train'], device, epochs=args.epochs,
                        max_train_samples=args.train_samples, ckpt_path=ck,
                        log_every=max(1, args.epochs // 3))

            h = evaluate(m, nr[train_on]['test'], device,
                         args.rollout_steps, args.eval_samples)
            a = evaluate(m, nr[away]['test'], device,
                         args.rollout_steps, args.eval_samples)
            ret = 100 * a['csi_final'] / h['csi_final'] if h['csi_final'] > 0 else float('nan')

            print(f"    home CSI {h['csi_final']:.4f} (vol {h['volume_ratio']:.2f}x)  "
                  f"away CSI {a['csi_final']:.4f} (vol {a['volume_ratio']:.2f}x)  "
                  f"retained {ret:.1f}%")
            results.append({'train_on': train_on, 'seed': seed,
                            'home': h, 'away': a, 'retained_pct': ret})

    print("\n  READ THIS: compare 'retained %' here with the WITH-rainfall runs.")
    print("  If transfer still fails without rainfall, the failure is about the")
    print("  flood mechanism, not about one missing input channel.")
    save('exp2_rainfall_ablation', results)
    return results


# ================================================================== #
# EXPERIMENT 4 — FINE-TUNING CURVE (independent events)
# ================================================================== #

def exp4_finetune_curve(data, device, args):
    """
    How many INDEPENDENT target events are needed?

    The reviewer's objection: 'are your 5 fine-tuning samples 5 separate
    floods, or 5 hours of the same flood?'

    Answer: for the breach data each sample IS a separate simulation
    (different ids), so events are independent by construction. We sweep
    0,1,2,5,10,20,40 events and also compare against training from scratch
    on the same number of events.
    """
    print("\n" + "=" * 66)
    print("  EXPERIMENT 4 — FINE-TUNING CURVE (independent events)")
    print("=" * 66)

    counts = args.ft_counts
    base, target = 'harvey', 'breach'   # breach samples are separate sims
    results = []

    for seed in args.seeds:
        print(f"\n  --- seed {seed}: pre-train on {base} ---")
        set_seed(seed)
        pre = build_model('vector', 64, 3, device)
        ck = os.path.join(OUT, f'ftbase_{base}_s{seed}.pt')
        train_model(pre, data[base]['train'], device, epochs=args.epochs,
                    max_train_samples=args.train_samples, ckpt_path=ck,
                    log_every=max(1, args.epochs // 3))
        base_state = {k: v.clone() for k, v in pre.state_dict().items()}

        zero = evaluate(pre, data[target]['test'], device,
                        args.rollout_steps, args.eval_samples)
        print(f"    zero-shot on {target}: CSI {zero['csi_final']:.4f}")
        results.append({'seed': seed, 'n_events': 0, 'mode': 'finetune',
                        'scores': zero})

        pool = data[target]['train']
        for n in counts:
            if n == 0 or n > len(pool):
                continue
            idx = np.random.RandomState(seed).choice(len(pool), n, replace=False)
            few = [pool[i] for i in idx]

            # --- fine-tune from the pre-trained weights ---
            m = build_model('vector', 64, 3, device)
            m.load_state_dict(base_state)
            train_model(m, few, device, epochs=args.ft_epochs, lr=1e-4,
                        max_train_samples=None, verbose=False)
            ft = evaluate(m, data[target]['test'], device,
                          args.rollout_steps, args.eval_samples)

            # --- train from scratch on the same n events ---
            set_seed(seed)
            sc = build_model('vector', 64, 3, device)
            train_model(sc, few, device, epochs=args.epochs, lr=1e-3,
                        max_train_samples=None, verbose=False)
            scr = evaluate(sc, data[target]['test'], device,
                           args.rollout_steps, args.eval_samples)

            print(f"    n={n:3d} events   fine-tuned CSI {ft['csi_final']:.4f}   "
                  f"from-scratch CSI {scr['csi_final']:.4f}")
            results.append({'seed': seed, 'n_events': n, 'mode': 'finetune',
                            'scores': ft})
            results.append({'seed': seed, 'n_events': n, 'mode': 'scratch',
                            'scores': scr})

    print("\n  READ THIS: the fine-tune curve vs the scratch curve answers")
    print("  'how much target data do you actually need, and does pre-training")
    print("  on another flood type help at all?'")
    save('exp4_finetune_curve', results)
    return results


# ================================================================== #
# EXPERIMENT 5 — STRONGER ARCHITECTURE
# ================================================================== #

def exp5_stronger_model(data, device, args):
    """
    Does the transfer failure survive a bigger network?

    The reviewer's objection: 'maybe transfer failed because your 42k-parameter
    model is too weak.'

    Answer: repeat with a substantially larger model. If the gap persists,
    it is not an artefact of model capacity.
    """
    print("\n" + "=" * 66)
    print("  EXPERIMENT 5 — STRONGER ARCHITECTURE")
    print("=" * 66)

    configs = [('small', 64, 3), ('large', 160, 6)]
    results = []

    for name, hidden, layers in configs:
        for train_on in ['breach', 'harvey']:
            away = 'harvey' if train_on == 'breach' else 'breach'
            for seed in args.seeds[:args.arch_seeds]:
                set_seed(seed)
                m = build_model('vector', hidden, layers, device)
                npar = sum(p.numel() for p in m.parameters())
                print(f"\n  --- {name} ({npar:,} params), train {train_on}, "
                      f"seed {seed} ---")
                ck = os.path.join(OUT, f'arch_{name}_{train_on}_s{seed}.pt')
                train_model(m, data[train_on]['train'], device,
                            epochs=args.epochs,
                            max_train_samples=args.train_samples,
                            ckpt_path=ck, log_every=max(1, args.epochs // 3))

                h = evaluate(m, data[train_on]['test'], device,
                             args.rollout_steps, args.eval_samples)
                a = evaluate(m, data[away]['test'], device,
                             args.rollout_steps, args.eval_samples)
                ret = (100 * a['csi_final'] / h['csi_final']
                       if h['csi_final'] > 0 else float('nan'))
                print(f"    home CSI {h['csi_final']:.4f}  away CSI "
                      f"{a['csi_final']:.4f}  retained {ret:.1f}%")
                results.append({'arch': name, 'params': npar,
                                'train_on': train_on, 'seed': seed,
                                'home': h, 'away': a, 'retained_pct': ret})

    print("\n  READ THIS: if the large model shows a similar retained %, the")
    print("  transfer failure is not a capacity problem.")
    save('exp5_stronger_model', results)
    return results


# ================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, choices=[1, 2, 4, 5], default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--ft-epochs', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--arch-seeds', type=int, default=2)
    ap.add_argument('--train-samples', type=int, default=200)
    ap.add_argument('--eval-samples', type=int, default=19)
    ap.add_argument('--rollout-steps', type=int, default=40)
    ap.add_argument('--ft-counts', type=int, nargs='+',
                    default=[1, 2, 5, 10, 20, 40])
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.ft_epochs = 2, 2
        args.seeds, args.arch_seeds = [0], 1
        args.train_samples, args.eval_samples = 4, 3
        args.rollout_steps, args.ft_counts = 5, [1, 2]

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(CACHE_PATH):
        build_cache()
    data = load_cache()
    print(f"Loaded cache: breach {len(data['breach']['train'])} train / "
          f"{len(data['breach']['test'])} test, harvey "
          f"{len(data['harvey']['train'])} train / {len(data['harvey']['test'])} test")

    todo = [args.only] if args.only else [1, 2, 4, 5]
    fns = {1: exp1_threshold_sweep, 2: exp2_rainfall_ablation,
           4: exp4_finetune_curve, 5: exp5_stronger_model}
    for e in todo:
        fns[e](data, device, args)

    print("\n" + "=" * 66)
    print("  SUITE COMPLETE")
    print("=" * 66)


if __name__ == '__main__':
    main()