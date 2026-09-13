"""Independently verify saved nested results and add explicitly descriptive subsets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from run_nested_validation import regression_metrics

ROOT = Path(__file__).resolve().parent
if ROOT.name == '02_CODIGO_NOTEBOOKS':
    ROOT = ROOT.parent
OUT = ROOT / '03_RESULTADOS' if (ROOT / '01_DATASET').exists() else ROOT / 'project_artifacts'


def main():
    run = json.loads((OUT / 'nested_run_metadata.json').read_text())
    data = ROOT / '01_DATASET' / 'barley_nir_long.npz' if (ROOT / '01_DATASET').exists() else OUT / 'barley_nir_long.npz'
    assert run['data_sha256'] == hashlib.sha256(data.read_bytes()).hexdigest()
    z = np.load(data, allow_pickle=True)
    y = z['day'].astype(int)
    dish = z['petri_dish'].astype(int)
    meta = pd.DataFrame({'petri_dish': dish, 'grain_id': z['grain_id'], 'stage': y,
                         'cultivar': z['cultivar'].astype(str),
                         'sample_group': z['sample_group'].astype(str)})
    assert not meta.duplicated(['petri_dish', 'grain_id', 'stage']).any()
    assert meta.groupby(['petri_dish', 'grain_id']).stage.nunique().eq(6).all()
    assert meta.groupby('petri_dish').sample_group.nunique().eq(1).all()
    oof = pd.read_csv(OUT / 'nested_oof_predictions.csv.gz')
    folds = pd.read_csv(OUT / 'nested_outer_fold_metrics.csv')
    inner = pd.read_csv(OUT / 'nested_inner_search.csv.gz')
    ci = pd.read_csv(OUT / 'nested_cluster_bootstrap_ci.csv')
    boot = pd.read_csv(OUT / 'nested_cluster_bootstrap_replicates.csv.gz')
    assert len(oof) == 3 * len(y) and len(folds) == 15 and len(inner) == 5 * 4 * 23
    assert len(boot) == 3 * 2000 and boot.groupby('family').bootstrap_repetition.nunique().eq(2000).all()
    expected_fold = np.empty(len(y), dtype=int)
    split_rows = []
    for number, (train_idx, test_idx) in enumerate(GroupKFold(5).split(y, y, dish), 1):
        assert set(dish[train_idx]).isdisjoint(dish[test_idx])
        expected_fold[test_idx] = number
        split_rows.append({'fold': number, 'train_dishes': len(np.unique(dish[train_idx])),
                           'test_dishes': len(np.unique(dish[test_idx])),
                           'train_observations': len(train_idx), 'test_observations': len(test_idx),
                           'train_test_dish_overlap': 0})
        for tr, va in GroupKFold(4).split(y[train_idx], y[train_idx], dish[train_idx]):
            assert set(dish[train_idx[tr]]).isdisjoint(dish[train_idx[va]])
    for family, frame in oof.groupby('family'):
        assert len(frame) == len(y) and frame.row_index.nunique() == len(y)
        index = frame.row_index.to_numpy(dtype=int)
        assert np.array_equal(frame.y_true.to_numpy(), y[index])
        assert np.array_equal(frame.petri_dish.to_numpy(), dish[index])
        assert np.array_equal(frame.outer_fold.to_numpy(), expected_fold[index])
        assert np.isfinite(frame.y_pred).all()
        recomputed = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        for metric, value in recomputed.items():
            row = ci.loc[ci.family.eq(family) & ci.metric.eq(metric)].iloc[0]
            assert np.isclose(value, row.estimate, rtol=1e-10, atol=1e-10)
            assert row.ci_low <= row.estimate <= row.ci_high
            sampled = boot.loc[boot.family.eq(family), metric]
            assert np.isclose(row.ci_low, sampled.quantile(0.025), rtol=1e-10, atol=1e-10)
            assert np.isclose(row.ci_high, sampled.quantile(0.975), rtol=1e-10, atol=1e-10)
        for fold, block in frame.groupby('outer_fold'):
            scores = regression_metrics(block.y_true.to_numpy(), block.y_pred.to_numpy())
            saved = folds.loc[folds.family.eq(family) & folds.outer_fold.eq(fold)].iloc[0]
            for metric, value in scores.items():
                assert np.isclose(value, saved[metric], rtol=1e-10, atol=1e-10)
            selected = inner.loc[inner.family.eq(family) & inner.outer_fold.eq(fold)]
            ranking = selected.groupby('candidate_key').agg(mae=('mae', 'mean'), rmse=('rmse', 'mean')).reset_index().sort_values(['mae', 'rmse', 'candidate_key'])
            assert ranking.iloc[0].candidate_key == saved.candidate_key
    assert folds.groupby('family').mae.mean().idxmin() == run['best_family']
    draws = boot.pivot(index='bootstrap_repetition', columns='family', values='mae')
    contrasts = pd.read_csv(OUT / 'nested_paired_mae_differences.csv')
    for row in contrasts.itertuples():
        alternative = row.comparison.split(' minus ')[0]
        difference = draws[alternative] - draws[run['best_family']]
        assert np.isclose(row.ci_low, difference.quantile(0.025), rtol=1e-10, atol=1e-10)
        assert np.isclose(row.ci_high, difference.quantile(0.975), rtol=1e-10, atol=1e-10)
    stress = pd.read_csv(OUT / 'domain_stress_test_summary.csv')
    stress_predictions = pd.read_csv(OUT / 'domain_stress_predictions.csv.gz')
    assert len(stress) == 6
    for _, row in stress.iterrows():
        field = 'cultivar' if row.scheme == 'leave_one_cultivar_out' else 'sample_group'
        test_mask = meta[field].eq(row.held_out)
        assert set(dish[~test_mask]).isdisjoint(dish[test_mask])
        frame = stress_predictions.loc[stress_predictions.scheme.eq(row.scheme) & stress_predictions.held_out.eq(row.held_out)]
        assert np.array_equal(np.sort(frame.row_index.to_numpy()), np.flatnonzero(test_mask))
        for metric, value in regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy()).items():
            assert np.isclose(row[metric], value, rtol=1e-10, atol=1e-10)
    best = oof.loc[oof.family.eq(run['best_family'])]
    subset_rows = []
    for name, frame in [('all_six_stages', best), ('post_moisture_1_to_5', best.loc[best.y_true.ge(1)])]:
        truth = frame.y_true.to_numpy()
        for predictor, prediction in [(run['best_family'], frame.y_pred.to_numpy()),
                                       ('constant_subset_mean', np.full(len(frame), truth.mean()))]:
            subset_rows.append({'subset': name, 'predictor': predictor, 'n': len(frame),
                                **regression_metrics(truth, prediction)})
    pd.DataFrame(subset_rows).to_csv(OUT / 'descriptive_stage_subsets.csv', index=False)
    pd.DataFrame(split_rows).to_csv(OUT / 'nested_split_audit.csv', index=False)
    audit = {'completed_run_signature': run['run_signature'], 'result': 'passed',
             'checks': ['dataset digest', 'compound keys and six-stage coverage', 'outer and inner dish disjointness',
                        'one external prediction per row and family', 'saved metric recalculation',
                        'inner selection ranking', '2000 paired bootstrap draws and interval quantiles per family',
                        'paired difference interval quantiles',
                        'six domain-test memberships and metric recalculation'],
             'subset_interpretation': 'Descriptive subsets of the same saved OOF predictions; no new fitting or independent test.'}
    (OUT / 'independent_result_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(subset_rows).to_string(index=False))


if __name__ == '__main__':
    main()
