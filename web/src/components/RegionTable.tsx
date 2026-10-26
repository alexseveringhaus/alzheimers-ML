import { useEffect, useEffectEvent, useRef } from 'react'

import { isValidSuvr, SUVR_MAX, SUVR_MIN, type FormValues, type Modality } from '../inputs'
import type { ShapContribution } from '../model/logreg'
import { PARAMS } from '../model/params'
import { prettyRoi } from '../roi'

interface Props {
  values: FormValues
  contributions: ShapContribution[] | null // null while any field is invalid
  onChange: (modality: Modality, roi: string, value: string) => void
  onReset: () => void
}

const STEP = 0.01
const n = PARAMS.amyloid_rois.length
const MEANS: Record<Modality, Record<string, number>> = {
  amyloid: Object.fromEntries(PARAMS.amyloid_rois.map((r, i) => [r, PARAMS.scaler_mean[i]])),
  tau: Object.fromEntries(PARAMS.tau_rois.map((r, i) => [r, PARAMS.scaler_mean[n + i]])),
}
const SUFFIX: Record<Modality, string> = { amyloid: '_amy', tau: '_tau' }

function Delta({ text, mean }: { text: string; mean: number }) {
  if (!isValidSuvr(text)) return <span className="delta" />
  const diff = Number(text) - mean
  if (Math.abs(diff) < 0.005) return <span className="delta">avg</span>
  return <span className={`delta ${diff > 0 ? 'up' : 'down'}`}>{diff > 0 ? '+' : '−'}{Math.abs(diff).toFixed(2)}</span>
}

/** One row per region with amyloid and tau side by side. */
export function RegionTable({ values, contributions, onChange, onReset }: Props) {
  const tableRef = useRef<HTMLTableElement>(null)
  const shap = new Map(contributions?.map((c) => [c.roi, c.shap_value]))
  const maxShap = Math.max(...(contributions ?? []).map((c) => Math.abs(c.shap_value)), 1e-9)
  const hasInvalid = [...Object.values(values.amyloid), ...Object.values(values.tau)].some((v) => !isValidSuvr(v))

  // Scrolling over a focused field steps its value. Chrome does this natively but
  // Safari doesn't, so handle it here for every browser (one step per wheel event).
  const onWheel = useEffectEvent((e: WheelEvent) => {
    const input = e.target
    if (!(input instanceof HTMLInputElement) || input !== document.activeElement || !input.dataset.modality) return
    e.preventDefault()
    if (e.deltaY === 0) return
    const modality = input.dataset.modality as Modality
    const roi = input.dataset.roi!
    const current = Number(values[modality][roi])
    if (!Number.isFinite(current)) return
    // Move to the next point on the 0.01 grid, as Chrome's own stepping does
    // (1.1766 -> 1.18 going up, -> 1.17 going down)
    const steps = current / STEP
    const target = e.deltaY < 0 ? Math.floor(steps + 1e-9) + 1 : Math.ceil(steps - 1e-9) - 1
    const next = Math.min(Math.max(target * STEP, SUVR_MIN), SUVR_MAX)
    onChange(modality, roi, next.toFixed(2))
  })
  useEffect(() => {
    const table = tableRef.current
    if (!table) return
    // Not passive, so preventDefault can stop the page from scrolling
    const listener = (e: WheelEvent) => onWheel(e)
    table.addEventListener('wheel', listener, { passive: false })
    return () => table.removeEventListener('wheel', listener)
  }, [])

  const cell = (modality: Modality, roi: string) => {
    const text = values[modality][roi]
    const value = shap.get(roi + SUFFIX[modality])
    const label = `${prettyRoi(roi)} ${modality}`
    return (
      <td className="num">
        <span className="value-cell">
          <input
            type="number"
            inputMode="decimal"
            name={`${modality}.${roi}`}
            data-modality={modality}
            data-roi={roi}
            min={SUVR_MIN}
            max={SUVR_MAX}
            step={STEP}
            value={text}
            aria-label={label}
            aria-invalid={!isValidSuvr(text)}
            onChange={(e) => onChange(modality, roi, e.target.value)}
          />
          <Delta text={text} mean={MEANS[modality][roi]} />
          <span
            className={`influence-dot ${value == null ? '' : value > 0 ? 'ad' : 'cn'}`}
            style={{ opacity: value == null ? 0 : 0.3 + 0.7 * Math.min(Math.abs(value) / maxShap, 1) }}
            title={value == null ? undefined : `Influence ${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`}
            aria-hidden="true"
          />
        </span>
      </td>
    )
  }

  return (
    <section className="panel region-panel" aria-labelledby="scan-values">
      <h2 id="scan-values">Scan values</h2>
      <p className="lede">
        Enter PET scan measurements for 13 brain regions from amyloid and tau scans. PET (Positron Emission Tomography)
        imaging measures protein buildup in the brain. Amyloid and tau are the two hallmark proteins that accumulate in
        Alzheimer's disease, and their regional levels (expressed as SUVRs, a normalized uptake ratio) are among the
        strongest known biomarkers for early detection.
      </p>
      <p className="lede">Values start at the training average. Change any of them and the result updates as you type.</p>
      <table className="region-table" ref={tableRef}>
        <thead>
          <tr>
            <th scope="col">Region</th>
            <th scope="col" className="num">Amyloid</th>
            <th scope="col" className="num">Tau</th>
          </tr>
        </thead>
        <tbody>
          {PARAMS.amyloid_rois.map((roi) => (
            <tr key={roi}>
              <th scope="row">{prettyRoi(roi)}</th>
              {cell('amyloid', roi)}
              {cell('tau', roi)}
            </tr>
          ))}
        </tbody>
      </table>
      {hasInvalid && (
        <p className="form-error" role="alert">
          Enter a value from {SUVR_MIN} to {SUVR_MAX} in every field.
        </p>
      )}
      <div className="table-foot">
        <p>Small numbers show the difference from the training average. Dots show each value's influence.</p>
        <button type="button" className="pill-button" onClick={onReset}>Reset to averages</button>
      </div>
      <p className="hint">Tip: click a value, then scroll to step it up or down.</p>
    </section>
  )
}
