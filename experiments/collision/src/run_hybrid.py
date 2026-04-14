# Experiment 1 extension: Leap Hybrid Solver on the full
# Shell 3 QUBO (N=1,656, k=100, sigma=0.1 km, Pc only).
# Adds Hybrid column to the SA vs SQA comparison.
# Requires: data/arnas_candidates.csv
# Output:   results/hybrid_comparison.csv

import numpy as np
import pandas as pd
import dimod
from dwave.system import LeapHybridSampler

df = pd.read_csv('data/arnas_candidates.csv')
N = len(df)
k = 100
P = 20000
pc = df['pc'].values

Q = {}
for i in range(N):
    Q[(i,i)] = P * (1 - 2*k)
    for j in range(i+1, N):
        w = (1 - pc[i]) * (1 - pc[j])
        Q[(i,j)] = -w + 2*P

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
sampler = LeapHybridSampler()
hybrid_results = []

for run in range(20):
    result = sampler.sample(bqm, time_limit=3)
    sample = result.first.sample
    selected = [i for i, v in sample.items() if v == 1]
    if len(selected) == k:
        agg_pc = 1.0 - np.prod(
            [1.0 - pc[i] for i in selected])
        hybrid_results.append({
            'run': run,
            'aggregate_pc': agg_pc,
            'n_selected': len(selected),
            'feasible': True
        })
    else:
        hybrid_results.append({
            'run': run,
            'aggregate_pc': np.nan,
            'n_selected': len(selected),
            'feasible': False
        })

results_df = pd.DataFrame(hybrid_results)
valid = results_df[results_df['feasible']]

print('=== Hybrid Solver Results (N=1656, k=100) ===')
print(f'Valid runs:        {len(valid)}/20')
print(f'Best aggregate Pc: {valid["aggregate_pc"].min():.3e}')
print(f'Mean aggregate Pc: {valid["aggregate_pc"].mean():.3e}')
print(f'Std:               {valid["aggregate_pc"].std():.3e}')

try:
    cmp = pd.read_csv('results/method_comparison.csv')
    random_mean = cmp.loc[
        cmp['method']=='random', 'mean_pc'].values[0]
    oom = (np.log10(random_mean) -
           np.log10(valid['aggregate_pc'].min()))
    print(f'OOM vs random (best): +{oom:.2f}')
except FileNotFoundError:
    print('Run analyze_results.py first to get baseline.')

results_df.to_csv('results/hybrid_comparison.csv', index=False)
print('Saved: results/hybrid_comparison.csv')
