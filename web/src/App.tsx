import params from '../../checkpoints/logreg_combined_params.json'
import { defaultInputs, predict, type ModelParams } from './model/logreg'

// Placeholder until the tabs are built: shows the prediction for the default inputs
export default function App() {
  const { amyloid, tau } = defaultInputs(params as ModelParams)
  const result = predict(params as ModelParams, amyloid, tau)
  return <p>{result.predicted_class}: P(AD) = {result.probability_ad.toFixed(3)}</p>
}
