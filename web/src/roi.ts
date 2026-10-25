const ROI_DISPLAY: Record<string, string> = {
  ROSTRALANTERIORCINGULATE: 'Rostral Anterior Cingulate',
  CAUDALANTERIORCINGULATE: 'Caudal Anterior Cingulate',
  MIDDLETEMPORAL: 'Middle Temporal',
  INFERIORTEMPORAL: 'Inferior Temporal',
  POSTERIORCINGULATE: 'Posterior Cingulate',
  PRECUNEUS: 'Precuneus',
  SUPERIORFRONTAL: 'Superior Frontal',
  SUPERIORPARIETAL: 'Superior Parietal',
  LATERALOCCIPITAL: 'Lateral Occipital',
  FRONTALPOLE: 'Frontal Pole',
  ENTORHINAL: 'Entorhinal',
  FUSIFORM: 'Fusiform',
  INFERIORPARIETAL: 'Inferior Parietal',
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
  const fallback = key.replaceAll('_', ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())
  return (ROI_DISPLAY[key] ?? fallback) + suffix
}
