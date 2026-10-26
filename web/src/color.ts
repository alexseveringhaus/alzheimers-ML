// Small color helpers for the brain map, which needs real hex values per region
// (CSS variables can't be interpolated along a multi-stop ramp).

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** Linear blend from color a (t = 0) to color b (t = 1). */
export function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a)
  const [r2, g2, b2] = hexToRgb(b)
  const channel = (x: number, y: number) => Math.round(x + (y - x) * t).toString(16).padStart(2, '0')
  return `#${channel(r1, r2)}${channel(g1, g2)}${channel(b1, b2)}`
}

/** Piecewise-linear color ramp through evenly spaced stops; t is clamped to [0, 1]. */
export function ramp(stops: string[], t: number): string {
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1)
  const i = Math.min(Math.floor(x), stops.length - 2)
  return mix(stops[i], stops[i + 1], x - i)
}
