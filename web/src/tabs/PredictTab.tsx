import { useMemo } from 'react'

import { ResultCard } from '../components/ResultCard'
import { RoiForm } from '../components/RoiForm'
import { ShapChart } from '../components/ShapChart'
import { parseForm, type FormValues, type Modality } from '../inputs'
import { predict } from '../model/logreg'
import { PARAMS } from '../model/params'

interface Props {
  values: FormValues
  onChange: (modality: Modality, roi: string, value: string) => void
  onReset: () => void
}

export function PredictTab({ values, onChange, onReset }: Props) {
  const result = useMemo(() => {
    const inputs = parseForm(values)
    return inputs ? predict(PARAMS, inputs.amyloid, inputs.tau) : null
  }, [values])

  return (
    <>
      <p className="intro">
        Enter PET scan measurements for 13 brain regions from <strong>amyloid</strong> and <strong>tau</strong> scans.
        PET (Positron Emission Tomography) imaging measures protein buildup in the brain. Amyloid and tau are the
        two hallmark proteins that accumulate in Alzheimer's disease, and their regional levels (expressed as SUVRs,
        a normalized uptake ratio) are among the strongest known biomarkers for early detection.
      </p>
      <p className="intro muted">
        The fields start at the training-set averages. The model runs in your browser, so the result updates as you
        type. The <a href="#api">Live API</a> tab sends the same inputs to the deployed server.
      </p>

      <div className="predict-layout">
        <RoiForm values={values} onChange={onChange} onReset={onReset} />

        <aside className="predict-result">
          {result ? (
            <>
              <ResultCard predictedClass={result.predicted_class} probabilityAd={result.probability_ad} />
              <h3>Which regions drove this prediction</h3>
              <ShapChart contributions={result.shap_contributions} />
              <p className="muted small">
                SHAP values for the top 15 of 26 features. Red bars push the prediction toward AD; blue bars push it
                toward CN. Bar length is each region's influence on this prediction, relative to the average subject.
              </p>
            </>
          ) : (
            <p className="muted">Fix the highlighted fields to see a prediction.</p>
          )}
        </aside>
      </div>
    </>
  )
}
