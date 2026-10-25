import type { ShapContribution } from '../model/logreg'
import { prettyRoi } from '../roi'

interface Props {
  contributions: ShapContribution[]  // sorted by absolute value, largest first
  limit?: number
}

/** Horizontal bars around zero: red pushes toward AD, blue toward CN. */
export function ShapChart({ contributions, limit = 15 }: Props) {
  const top = contributions.slice(0, limit)
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.shap_value)), 1e-9)

  return (
    <div className="shap">
      <div className="shap-axis-labels" aria-hidden="true">
        <span>← toward CN</span>
        <span>toward AD →</span>
      </div>
      <ol className="shap-rows">
        {top.map((c) => {
          const width = `${(Math.abs(c.shap_value) / maxAbs) * 50}%`
          const positive = c.shap_value > 0
          const label = prettyRoi(c.roi)
          const value = `${positive ? '+' : ''}${c.shap_value.toFixed(3)}`
          return (
            <li key={c.roi} className="shap-row" aria-label={`${label}: ${value}, ${positive ? 'toward AD' : 'toward CN'}`}>
              <span className="shap-label">{label}</span>
              <span className="shap-track" aria-hidden="true">
                <span
                  className={`shap-bar ${positive ? 'ad' : 'cn'}`}
                  style={positive ? { left: '50%', width } : { right: '50%', width }}
                />
              </span>
              <span className="shap-value" aria-hidden="true">{value}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
