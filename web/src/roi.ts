const ROI_DISPLAY: Record<string, string> = {
  ROSTRALANTERIORCINGULATE: 'Rostral anterior cingulate',
  CAUDALANTERIORCINGULATE: 'Caudal anterior cingulate',
  MIDDLETEMPORAL: 'Middle temporal',
  INFERIORTEMPORAL: 'Inferior temporal',
  POSTERIORCINGULATE: 'Posterior cingulate',
  PRECUNEUS: 'Precuneus',
  SUPERIORFRONTAL: 'Superior frontal',
  SUPERIORPARIETAL: 'Superior parietal',
  LATERALOCCIPITAL: 'Lateral occipital',
  FRONTALPOLE: 'Frontal pole',
  ENTORHINAL: 'Entorhinal',
  FUSIFORM: 'Fusiform',
  INFERIORPARIETAL: 'Inferior parietal',
}

/**
 * Convert a raw ROI key to a readable label.
 * Handles plain names (CTX_ENTORHINAL_SUVR) and SHAP feature names that carry
 * a modality suffix (CTX_ENTORHINAL_SUVR_amy / _tau).
 */
export function prettyRoi(name: string): string {
  let suffix = ''
  if (name.endsWith('_amy')) {
    suffix = ' (Amy)'
    name = name.slice(0, -4)
  } else if (name.endsWith('_tau')) {
    suffix = ' (Tau)'
    name = name.slice(0, -4)
  }
  const key = name.replace('CTX_', '').replace('_SUVR', '')
  const words = key.replaceAll('_', ' ').toLowerCase()
  const fallback = words.charAt(0).toUpperCase() + words.slice(1)
  return (ROI_DISPLAY[key] ?? fallback) + suffix
}

/** Region key used by the brain map: CTX_ENTORHINAL_SUVR(_amy|_tau) -> ENTORHINAL */
export function regionKey(name: string): string {
  return name.replace(/_(amy|tau)$/, '').replace('CTX_', '').replace('_SUVR', '')
}

/** Sentence-style feature label: CTX_ENTORHINAL_SUVR_tau -> "Entorhinal tau" */
export function featureLabel(name: string): string {
  const tracer = name.endsWith('_amy') ? ' amyloid' : name.endsWith('_tau') ? ' tau' : ''
  return prettyRoi(name.replace(/_(amy|tau)$/, '')) + tracer
}
