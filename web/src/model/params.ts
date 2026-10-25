import rawParams from '../../../checkpoints/logreg_combined_params.json'
import type { ModelParams } from './logreg'

/** The exported combined logistic regression (see LogisticRegression/export_params.py). */
export const PARAMS = rawParams as ModelParams
