interface Props {
  predictedClass: 'AD' | 'CN'
  probabilityAd: number
}

const CLASS_NAMES = { AD: "Alzheimer's Disease", CN: 'Cognitively Normal' }

const pct = (p: number) => `${(p * 100).toFixed(1)}%`

export function ResultCard({ predictedClass, probabilityAd }: Props) {
  return (
    <div className="result-card" aria-live="polite">
      <div className="result-class">
        <span className={`class-badge ${predictedClass.toLowerCase()}`}>{predictedClass}</span>
        <span className="result-class-name">{CLASS_NAMES[predictedClass]}</span>
      </div>
      <div className="prob">
        <div className="prob-labels">
          <span>AD {pct(probabilityAd)}</span>
          <span>CN {pct(1 - probabilityAd)}</span>
        </div>
        <div
          className="prob-track"
          role="meter"
          aria-label="Probability of Alzheimer's Disease"
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={probabilityAd}
        >
          <div className="prob-fill" style={{ width: pct(probabilityAd) }} />
          <div className="prob-threshold" title="Decision threshold (50%)" />
        </div>
      </div>
    </div>
  )
}
