import { useState } from 'react'

import type { ShapContribution } from '../model/logreg'
import { featureLabel } from '../roi'

interface Props {
  contributions: ShapContribution[] // sorted by absolute value, largest first
  initial?: number
}

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

/** Diverging bars around zero: toward AD to the right, toward CN to the left. */
export function InfluenceBars({ contributions, initial = 8 }: Props) {
  const [showAll, setShowAll] = useState(false)
  const shown = showAll ? contributions : contributions.slice(0, initial)
  const max = Math.max(Math.abs(contributions[0]?.shap_value ?? 0), 1e-9)

  return (
    <div className="influences">
      <h2>Largest influences</h2>
      <ol className="influence-list">
        {shown.map((c) => {
          const toAd = c.shap_value > 0
          const name = featureLabel(c.roi)
          return (
            <li key={c.roi} aria-label={`${name}: ${signed(c.shap_value)}, toward ${toAd ? 'AD' : 'CN'}`}>
              <span className="influence-name">{name}</span>
              <span className="influence-track" aria-hidden="true">
                <span className={`influence-bar ${toAd ? 'ad' : 'cn'}`} style={{ width: `${(Math.abs(c.shap_value) / max) * 50}%` }} />
              </span>
              <span className="influence-value" aria-hidden="true">{signed(c.shap_value)}</span>
            </li>
          )
        })}
      </ol>
      {contributions.length > initial && (
        <button type="button" className="text-button" onClick={() => setShowAll((v) => !v)}>
          {showAll ? `Show top ${initial}` : `Show all ${contributions.length}`}
        </button>
      )}
    </div>
  )
}
