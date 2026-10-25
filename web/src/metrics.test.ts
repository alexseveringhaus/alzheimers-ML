import { describe, expect, it } from 'vitest'

import metrics from '../../results/metrics.json'
import { collectMetricRows } from './metrics'

describe('collectMetricRows', () => {
  it('reads nested and flat entries from metrics.json', () => {
    const rows = collectMetricRows(metrics)
    const models = rows.map((r) => r.model)
    expect(models).toContain('NeuralNetwork')          // flat entry
    expect(models).toContain('LogisticRegression')     // nested under feature set
    expect(models).toContain('Baseline + slope')       // nested under LongitudinalTau
  })

  it('formats metrics and handles missing AUC', () => {
    const rows = collectMetricRows({
      CNN: { feature_set: 'slices', cv_auc_mean: null, cv_auc_std: null, test_accuracy: 0.8125, test_auc: null },
      combined: { LogisticRegression: { feature_set: 'combined', cv_auc_mean: 0.892, cv_auc_std: 0.0606, test_accuracy: 0.9867, test_auc: 1 } },
    })
    expect(rows).toEqual([
      { model: 'CNN', featureSet: 'slices', cvAuc: 'N/A', testAuc: 'N/A', testAccuracy: '81.3%' },
      { model: 'LogisticRegression', featureSet: 'combined', cvAuc: '0.892 ± 0.061', testAuc: '1.000', testAccuracy: '98.7%' },
    ])
  })
})
