import { describe, expect, it } from 'vitest'

import metrics from '../../results/metrics.json'
import { collectMetricRows } from './metrics'

describe('collectMetricRows', () => {
  it('reads nested and flat entries from metrics.json, with readable names', () => {
    const rows = collectMetricRows(metrics)
    const models = rows.map((r) => r.model)
    expect(models).toContain('Neural network')       // flat entry
    expect(models).toContain('Logistic regression')  // nested under feature set
    expect(models).toContain('Longitudinal LR')      // nested under LongitudinalTau
    expect(models).toContain('2D CNN')               // flat, with no AUC values
    expect(rows.find((r) => r.model === 'Longitudinal LR')?.featureSet).toBe('Baseline tau')
  })

  it('marks only the combined logistic regression as deployed', () => {
    const deployed = collectMetricRows(metrics).filter((r) => r.deployed)
    expect(deployed).toHaveLength(1)
    expect(deployed[0]).toMatchObject({ model: 'Logistic regression', featureSet: 'Combined' })
  })

  it('formats metrics and handles missing AUC', () => {
    const rows = collectMetricRows({
      CNN: { feature_set: 'slices', cv_auc_mean: null, cv_auc_std: null, test_accuracy: 0.8125, test_auc: null },
      combined: { LogisticRegression: { feature_set: 'combined', cv_auc_mean: 0.892, cv_auc_std: 0.0606, test_accuracy: 0.9867, test_auc: 1 } },
    })
    expect(rows).toEqual([
      { model: '2D CNN', featureSet: 'PET image slices', cvAuc: 'N/A', testAuc: 'N/A', testAccuracy: '81.3%', deployed: false, cvAucMean: null },
      { model: 'Logistic regression', featureSet: 'Combined', cvAuc: '0.892 ± 0.061', testAuc: '1.000', testAccuracy: '98.7%', deployed: true, cvAucMean: 0.892 },
    ])
  })
})
