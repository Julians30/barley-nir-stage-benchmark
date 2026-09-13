"""Independently recalculate saved review metrics; no model fitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def main(root=None):
    root = Path(root or __file__).resolve()
    if root.is_file():
        root = root.parent
    if root.name == '02_CODIGO_NOTEBOOKS':
        root = root.parent
    data_path = root / '01_DATASET' / 'barley_nir_long.npz'
    results = root / '03_RESULTADOS'
    if not data_path.exists():
        results = root / 'project_artifacts'
        data_path = results / 'barley_nir_long.npz'
    revision = results / 'revision_r1'
    audit = json.loads((revision / 'review_additions_audit.json').read_text())
    data = np.load(data_path, allow_pickle=True)
    y, dish = data['day'], data['petri_dish']
    folds = list(GroupKFold(5).split(y, y, dish))
    original = pd.read_csv(results / 'nested_oof_predictions.csv.gz')
    original = original.loc[original.family.eq('SVR_RBF')].sort_values('row_index')
    assert audit['primary_oof_sha256'] == hashlib.sha256((results / 'nested_oof_predictions.csv.gz').read_bytes()).hexdigest()
    frozen = pd.read_csv(revision / 'svr_frozen_C_predictions.csv.gz')
    summary = pd.read_csv(revision / 'svr_frozen_C_summary.csv')
    recorded_folds = pd.read_csv(revision / 'svr_frozen_C_fold_metrics.csv')
    metric_checks = 0
    def check(values, expected):
        nonlocal metric_checks
        actual = {'mae': mean_absolute_error(values.y_true, values.y_pred),
                  'rmse': mean_squared_error(values.y_true, values.y_pred)**.5,
                  'r2': r2_score(values.y_true, values.y_pred),
                  'bias': (values.y_pred-values.y_true).mean(),
                  'within_1_day': np.mean(np.abs(values.y_pred-values.y_true) <= 1)}
        for key, value in actual.items():
            np.testing.assert_allclose(value, expected[key], atol=1e-10)
            metric_checks += 1
    assert set(frozen.C.unique()) == {100, 300, 1000}
    for c_value, frame in frozen.groupby('C'):
        assert frame.parameter.str.startswith(f'C={c_value};').all()
        assert frame.primary_selected_parameter.str.startswith('C=100;').all()
        assert len(frame) == len(y) and frame.row_index.nunique() == len(y)
        np.testing.assert_array_equal(frame.sort_values('row_index').y_true, y)
        for subset in ['all_six_stages', 'post_moisture_1_to_5']:
            values = frame if subset == 'all_six_stages' else frame.loc[frame.y_true.ge(1)]
            expected = summary.loc[summary.C.eq(c_value) & summary.subset.eq(subset)].iloc[0]
            assert len(values) == expected['n']
            check(values, expected)
        for fold, (train, test) in enumerate(folds, 1):
            assert set(dish[train]).isdisjoint(set(dish[test]))
            values = frame.loc[frame.outer_fold.eq(fold)].sort_values('row_index')
            np.testing.assert_array_equal(values.row_index, test)
            check(values, recorded_folds.loc[recorded_folds.C.eq(c_value) & recorded_folds.outer_fold.eq(fold)].iloc[0])
    base = frozen.loc[frozen.C.eq(100)].sort_values('row_index')
    np.testing.assert_allclose(base.y_pred, original.y_pred, atol=1e-12)
    all_mae = np.abs(base.y_pred-base.y_true).mean()
    dry_mae = np.abs(base.loc[base.y_true.eq(0), 'y_pred']).mean()
    post = base.loc[base.y_true.ge(1)]
    post_mae = np.abs(post.y_pred-post.y_true).mean()
    np.testing.assert_allclose(all_mae, (dry_mae+5*post_mae)/6, atol=1e-12)
    profiles = pd.read_csv(revision / 'pls_training_profiles.csv')
    assert len(profiles) == 2040
    for (fold, subset), group in profiles.groupby(['outer_fold', 'training_subset']):
        assert len(group) == 204 and group.channel_index_0based.nunique() == 204
        train = folds[int(fold)-1][0]
        n = len(train) if subset == 'all_six_stages' else np.sum(y[train] >= 1)
        assert group.n_training_observations.eq(n).all()
        assert np.isfinite(group[['vip', 'coefficient_per_training_sd']]).all().all()
        np.testing.assert_allclose(np.mean(group.vip**2), 1, atol=1e-10)
        np.testing.assert_allclose(group.wavelength_nm_approx, np.linspace(950,1650,204), atol=1e-10)
    bands = pd.read_csv(revision / 'pls_band_summaries.csv')
    for row in bands.itertuples():
        low, high = map(float, row.band_nm_approx.split('-'))
        values = profiles.loc[profiles.outer_fold.eq(row.outer_fold) & profiles.training_subset.eq(row.training_subset)
            & profiles.wavelength_nm_approx.between(low,high)]
        assert len(values) == row.n_channels
        for actual, expected in [(values.vip.mean(),row.vip_mean),(values.vip.max(),row.vip_max),
             (values.vip.gt(1).mean(),row.fraction_vip_gt_1),
             (values.coefficient_per_training_sd.abs().mean(),row.coefficient_abs_mean)]:
            np.testing.assert_allclose(actual,expected,atol=1e-10)
    report = {'result': 'passed', 'primary_run_signature': audit['primary_run_signature'],
        'review_version': audit['version'], 'metric_checks': metric_checks,
        'n_original_rows': len(y), 'n_frozen_C_rows': len(frozen), 'n_PLS_profile_rows': len(profiles),
        'C100_equals_original_predictions': True, 'outer_membership_matches_primary': True,
        'all_stage_mae': float(all_mae), 'post_moisture_mae': float(post_mae),
        'dry_reference_mae': float(dry_mae), 'aggregation_reduction_percent': float(100*(1-all_mae/post_mae)),
        'scope': 'Recalculation and partition/profile provenance checks of saved files; no causal validation or model refitting.'}
    (revision / 'independent_review_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
