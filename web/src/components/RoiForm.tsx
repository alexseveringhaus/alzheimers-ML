import { useId } from 'react'

import { isValidSuvr, SUVR_MAX, SUVR_MIN, type FormValues, type Modality } from '../inputs'
import { prettyRoi } from '../roi'

interface Props {
  values: FormValues
  onChange: (modality: Modality, roi: string, value: string) => void
  onReset: () => void
}

const SECTIONS: { modality: Modality; title: string }[] = [
  { modality: 'amyloid', title: 'Amyloid PET SUVRs' },
  { modality: 'tau', title: 'Tau PET SUVRs' },
]

export function RoiForm({ values, onChange, onReset }: Props) {
  // Two tabs render this form at once, so input ids must be unique per instance
  const formId = useId()
  const hasInvalid = [...Object.values(values.amyloid), ...Object.values(values.tau)].some((v) => !isValidSuvr(v))

  return (
    <form className="roi-form" onSubmit={(e) => e.preventDefault()}>
      <div className="roi-columns">
        {SECTIONS.map(({ modality, title }) => (
          <fieldset key={modality}>
            <legend>{title}</legend>
            {Object.entries(values[modality]).map(([roi, value]) => {
              const id = `${formId}-${modality}-${roi}`
              const invalid = !isValidSuvr(value)
              return (
                <div className="roi-field" key={roi}>
                  <label htmlFor={id}>{prettyRoi(roi)}</label>
                  <input
                    id={id}
                    name={`${modality}.${roi}`}
                    type="number"
                    inputMode="decimal"
                    min={SUVR_MIN}
                    max={SUVR_MAX}
                    step={0.01}
                    value={value}
                    aria-invalid={invalid}
                    onChange={(e) => onChange(modality, roi, e.target.value)}
                  />
                </div>
              )
            })}
          </fieldset>
        ))}
      </div>
      <div className="form-footer">
        <button type="button" className="secondary" onClick={onReset}>
          Reset to averages
        </button>
        {hasInvalid && (
          <p className="form-error" role="alert">
            Enter a value from {SUVR_MIN} to {SUVR_MAX} in every field.
          </p>
        )}
      </div>
    </form>
  )
}
