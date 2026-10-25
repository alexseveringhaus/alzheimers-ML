import { describe, expect, it } from 'vitest'

import params from '../../../checkpoints/logreg_combined_params.json'
import fixtures from './fixtures.json'
import {
  defaultInputs,
  predict,
  PROB_TOLERANCE as PROB_TOL,
  SHAP_TOLERANCE as SHAP_TOL,
  type ModelParams,
} from './logreg'

// fixtures.json comes from scripts/make_parity_fixtures.py, which runs the API's own predict()

describe('browser logistic regression matches the API', () => {
  it('has fixtures for both classes', () => {
    const classes = new Set(fixtures.map((f) => f.expected.predicted_class))
    expect(classes).toEqual(new Set(['AD', 'CN']))
  })

  fixtures.forEach((fixture, i) => {
    it(`fixture ${i}`, () => {
      const result = predict(params as ModelParams, fixture.amyloid_suvrs, fixture.tau_suvrs)
      const expected = fixture.expected

      expect(result.predicted_class).toBe(expected.predicted_class)
      expect(Math.abs(result.probability_ad - expected.probability_ad)).toBeLessThanOrEqual(PROB_TOL)
      expect(Math.abs(result.probability_cn - expected.probability_cn)).toBeLessThanOrEqual(PROB_TOL)

      expect(result.shap_contributions.map((c) => c.roi)).toEqual(expected.shap_contributions.map((c) => c.roi))
      result.shap_contributions.forEach((c, j) => {
        const e = expected.shap_contributions[j]
        expect(Math.abs(c.shap_value - e.shap_value)).toBeLessThanOrEqual(SHAP_TOL)
        expect(c.direction).toBe(e.direction)
      })
    })
  })
})

describe('defaultInputs', () => {
  it('uses the training means, rounded to 4 decimals', () => {
    const { amyloid, tau } = defaultInputs(params as ModelParams)
    // Same value as AMY_DEFAULTS in the old Streamlit dashboard
    expect(amyloid.CTX_ENTORHINAL_SUVR).toBe(0.9235)
    expect(Object.keys(amyloid)).toHaveLength(13)
    expect(Object.keys(tau)).toHaveLength(13)
  })
})
