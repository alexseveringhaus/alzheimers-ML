import { useId, useState } from 'react'

import { mix, ramp } from '../color'
import { prettyRoi } from '../roi'
import { useTheme } from '../theme'

/** Region key (e.g. ENTORHINAL) -> SHAP value; positive pushes toward AD. */
export type RegionValues = Record<string, number>

interface Props {
  amyloid: RegionValues
  tau: RegionValues
}

type Tracer = 'both' | 'amyloid' | 'tau'

// ── Geometry ──────────────────────────────────────────────────────────────
// A simplified diagram, not exact anatomy: one hemisphere seen from the outside
// (lateral) and from the inside (medial), front facing left. Measured regions
// are shapes clipped to the cerebrum outline; unmeasured cortex stays neutral.

const VIEW_W = 380
const OUTLINE =
  'M 60 150 C 40 100 80 40 160 28 C 240 16 330 50 350 120 C 360 150 345 175 320 180 ' +
  'C 300 190 280 192 262 190 C 250 214 200 228 150 222 C 118 218 100 204 98 188 ' +
  'C 96 176 104 166 116 162 C 98 162 74 162 60 150 Z'
const CEREBELLUM = 'M 236 196 C 250 176 330 172 340 200 C 346 222 300 236 268 230 C 248 226 232 214 236 196 Z'
const BRAINSTEM = 'M 232 200 C 240 214 244 236 246 260 L 270 260 C 266 236 262 214 258 198 Z'
const FRONTAL_POLE = '0,0 102,0 98,36 88,80 84,120 90,166 0,260'

const LATERAL: Record<string, string> = {
  FRONTALPOLE: FRONTAL_POLE,
  SUPERIORFRONTAL: '102,0 198,0 196,20 194,52 160,58 124,70 86,86 88,80 98,36',
  SUPERIORPARIETAL: '198,0 330,0 318,40 312,90 310,96 270,82 230,70 190,60 194,52 196,20',
  INFERIORPARIETAL: '190,60 230,70 270,82 310,96 304,140 300,150 250,122 210,126 172,140 176,110 184,80',
  LATERALOCCIPITAL: '330,0 380,0 380,260 290,260 296,200 304,140 312,90 318,40',
  MIDDLETEMPORAL: '110,184 150,168 200,156 250,148 300,150 300,178 270,176 220,182 170,192 120,206 100,200',
  INFERIORTEMPORAL: '100,200 120,206 170,192 220,182 270,176 300,178 296,260 80,260',
}
const MEDIAL: Record<string, string> = {
  FRONTALPOLE: FRONTAL_POLE,
  SUPERIORFRONTAL: '102,0 232,0 236,60 210,58 160,66 120,84 100,120 86,136 84,120 88,80 98,36',
  PRECUNEUS: '232,0 330,0 318,40 312,90 304,124 282,104 254,76 236,60',
  ENTORHINAL: '96,190 150,196 160,260 80,260',
  FUSIFORM: '150,196 300,186 296,260 160,260',
}
// The cingulate gyrus wraps around the corpus callosum, drawn as three thick arcs
const CINGULATE: Record<string, string> = {
  ROSTRALANTERIORCINGULATE: 'M 114 152 C 110 126 122 106 142 96',
  CAUDALANTERIORCINGULATE: 'M 147 94 C 172 82 206 78 233 82',
  POSTERIORCINGULATE: 'M 238 83 C 268 90 292 108 298 138',
}
const CORPUS_CALLOSUM = 'M 132 142 C 150 106 252 98 280 136'

// ── Colors ────────────────────────────────────────────────────────────────
// Toward AD follows the "hot" scale PET viewers use; toward CN is a cool blue.
// t is the region's share of the largest influence on screen (0 to 1).

interface Palette {
  neutral: string
  gap: string
  hind: string
  callosum: string
  ad: (t: number) => string
  cn: (t: number) => string
}

const LIGHT: Palette = {
  neutral: '#DCE3EC',
  gap: '#FFFFFF',
  hind: '#E9EEF4',
  callosum: '#C6D0DC',
  ad: (t) => mix('#F3E3DA', ramp(['#F2B13E', '#E8613C', '#C2253F'], t), 0.25 + 0.75 * t),
  cn: (t) => mix('#DCE8F2', '#1E7FB8', 0.25 + 0.75 * t),
}

const DARK: Palette = {
  neutral: '#24344F',
  gap: '#172540',
  hind: '#1B2840',
  callosum: '#31435F',
  ad: (t) => ramp(['#3A3142', '#F6D98A', '#F2B13E', '#E8613C', '#C2253F'], 0.2 + 0.8 * t),
  cn: (t) => mix('#24344F', '#57C7E8', 0.3 + 0.7 * t),
}

const TRACERS: { id: Tracer; label: string }[] = [
  { id: 'both', label: 'Both' },
  { id: 'amyloid', label: 'Amyloid' },
  { id: 'tau', label: 'Tau' },
]

const label = (region: string) => prettyRoi(`CTX_${region}_SUVR`)
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

export function BrainMap({ amyloid, tau }: Props) {
  const [tracer, setTracer] = useState<Tracer>('both')
  const palette = useTheme() === 'dark' ? DARK : LIGHT
  const clipId = useId()

  const regions = Object.keys(amyloid)
  const values: RegionValues =
    tracer === 'amyloid' ? amyloid
    : tracer === 'tau' ? tau
    : Object.fromEntries(regions.map((r) => [r, amyloid[r] + tau[r]]))
  // Single-tracer views share one scale, so switching between them compares fairly
  const scaleMax =
    tracer === 'both'
      ? Math.max(...Object.values(values).map(Math.abs))
      : Math.max(...regions.flatMap((r) => [Math.abs(amyloid[r]), Math.abs(tau[r])]))
  const max = Math.max(scaleMax, 1e-9)

  const fill = (region: string) => {
    const v = values[region] ?? 0
    const t = Math.min(Math.abs(v) / max, 1)
    return v >= 0 ? palette.ad(t) : palette.cn(t)
  }
  const tip = (region: string) => `${label(region)}: ${signed(values[region] ?? 0)}`

  const top = [...regions].sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]))[0]
  const summary = `Brain regions shaded by influence. Strongest: ${label(top)}, ${values[top] > 0 ? 'toward AD' : 'toward CN'}.`

  const view = (key: string, shapes: Record<string, string>, caption: string, x: number, medial: boolean) => (
    <g transform={`translate(${x},0)`}>
      <path d={BRAINSTEM} style={{ fill: palette.hind }} />
      <path d={CEREBELLUM} style={{ fill: palette.hind, stroke: palette.gap }} strokeWidth={3} />
      <clipPath id={`${clipId}-${key}`}>
        <path d={OUTLINE} />
      </clipPath>
      <g clipPath={`url(#${clipId}-${key})`}>
        <path d={OUTLINE} style={{ fill: palette.neutral }} />
        {Object.entries(shapes).map(([region, points]) => (
          <polygon key={region} className="region" points={points} style={{ fill: fill(region), stroke: palette.gap }} strokeWidth={3} strokeLinejoin="round">
            <title>{tip(region)}</title>
          </polygon>
        ))}
        {medial && (
          <>
            <path d={CORPUS_CALLOSUM} fill="none" style={{ stroke: palette.callosum }} strokeWidth={14} strokeLinecap="round" />
            {Object.entries(CINGULATE).map(([region, d]) => (
              <path key={region} className="region" d={d} fill="none" style={{ stroke: fill(region) }} strokeWidth={22}>
                <title>{tip(region)}</title>
              </path>
            ))}
          </>
        )}
      </g>
      <path d={OUTLINE} fill="none" style={{ stroke: palette.gap }} strokeWidth={3} />
      <text x={VIEW_W / 2} y={274} textAnchor="middle" className="brain-caption">{caption}</text>
    </g>
  )

  return (
    <div className="brain-map">
      <div className="panel-head">
        <h2>Where the signal comes from</h2>
        <div className="segmented" role="group" aria-label="Tracer shown on the map">
          {TRACERS.map((t) => (
            <button key={t.id} type="button" aria-pressed={tracer === t.id} onClick={() => setTracer(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <svg className="brain" viewBox={`0 0 ${VIEW_W * 2 + 16} 284`} role="img" aria-label={summary}>
        {view('lateral', LATERAL, 'Outer surface', 0, false)}
        {view('medial', MEDIAL, 'Inner surface', VIEW_W + 16, true)}
      </svg>
      <div className="heat-legend" aria-hidden="true">
        <span>Toward CN</span>
        <span className="heat-bar" />
        <span>Toward AD</span>
      </div>
    </div>
  )
}
