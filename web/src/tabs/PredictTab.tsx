import { useEffect, useMemo, useRef, useState } from 'react'

import { BrainMap, type RegionValues } from '../components/BrainMap'
import { InfluenceBars } from '../components/InfluenceBars'
import { RegionTable } from '../components/RegionTable'
import { Verdict } from '../components/Verdict'
import { parseForm, type FormValues, type Modality } from '../inputs'
import { predict } from '../model/logreg'
import { PARAMS } from '../model/params'
import { regionKey } from '../roi'

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

  // Split the 26 SHAP values into one map per tracer, keyed by region
  const byTracer = useMemo(() => {
    if (!result) return null
    const amyloid: RegionValues = {}
    const tau: RegionValues = {}
    for (const c of result.shap_contributions) {
      ;(c.roi.endsWith('_amy') ? amyloid : tau)[regionKey(c.roi)] = c.shap_value
    }
    return { amyloid, tau }
  }, [result])

  // On narrow screens a compact result is pinned to the top while editing; hide it
  // once the full result card is on screen, so the result isn't shown twice
  const verdictRef = useRef<HTMLElement>(null)
  const [verdictVisible, setVerdictVisible] = useState(false)
  useEffect(() => {
    const el = verdictRef.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => setVerdictVisible(entry.isIntersecting))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <>
      <section className="motivation" aria-labelledby="motivation-title">
        <h2 id="motivation-title">Why I built this</h2>
        <p>
          Dementia affects over 55 million people worldwide, and Alzheimer's disease, its most common cause, currently has
          no cure. Early and accurate diagnosis is critical for treatment planning and clinical trial enrollment. This
          project was driven by a genuine interest and personal connection to the problem — using ML on real clinical
          imaging data felt like a meaningful way to develop applied ML skills while working on something that I'm
          passionate about.
        </p>
      </section>
      {result && (
        <div className={verdictVisible ? 'pinned-result hidden' : 'pinned-result'} aria-hidden="true">
          <Verdict probabilityAd={result.probability_ad} compact />
        </div>
      )}
      <div className="predict-layout">
        <RegionTable values={values} contributions={result?.shap_contributions ?? null} onChange={onChange} onReset={onReset} />

        <aside className="predict-side" aria-label="Prediction">
          <section className="panel" aria-live="polite" ref={verdictRef}>
            {result ? (
              <Verdict probabilityAd={result.probability_ad} />
            ) : (
              <p className="empty-state">Fix the highlighted values to see a prediction.</p>
            )}
          </section>
          {result && byTracer && (
            <section className="panel">
              <BrainMap amyloid={byTracer.amyloid} tau={byTracer.tau} />
              <InfluenceBars contributions={result.shap_contributions} />
            </section>
          )}
          <p className="fine-print">
            The model runs in your browser. Values are SHAP contributions in log-odds, relative to the average subject.
            Research tool trained on ADNI data; not validated for clinical use.
          </p>
        </aside>
      </div>
    </>
  )
}
