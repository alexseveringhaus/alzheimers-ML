interface Props {
  probabilityAd: number
  compact?: boolean
}

function headline(p: number): string {
  if (p < 0.2) return 'Looks cognitively normal'
  if (p < 0.5) return 'Leans toward cognitively normal'
  if (p < 0.8) return "Leans toward Alzheimer's disease"
  return "Looks like Alzheimer's disease"
}

function percent(p: number): string {
  if (p < 0.005) return 'under 1%'
  if (p > 0.995) return 'over 99%'
  return `${Math.round(p * 100)}%`
}

/** Headline plus a heat scale with a knob at P(AD); the gap in the scale marks the 50% threshold. */
export function Verdict({ probabilityAd, compact = false }: Props) {
  return (
    <div className={compact ? 'verdict compact' : 'verdict'}>
      <p className="verdict-headline">
        {headline(probabilityAd)}
        {compact && <span className="verdict-percent">{percent(probabilityAd)} AD</span>}
      </p>
      <div
        className="heat-scale"
        role="meter"
        aria-label="Probability of Alzheimer's disease"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={probabilityAd}
        aria-valuetext={percent(probabilityAd)}
      >
        <span className="heat-threshold" />
        <span className="heat-knob" style={{ left: `${probabilityAd * 100}%` }} />
      </div>
      {!compact && (
        <>
          <div className="heat-ends" aria-hidden="true">
            <span>Cognitively normal</span>
            <span>Alzheimer's</span>
          </div>
          <p className="verdict-sentence">
            The model gives these values a <strong>{percent(probabilityAd)}</strong> probability of Alzheimer's disease.
          </p>
        </>
      )}
    </div>
  )
}
