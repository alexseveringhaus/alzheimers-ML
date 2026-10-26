import { describe, expect, it } from 'vitest'

import { mix, ramp } from './color'

describe('mix', () => {
  it('returns the end colors at t = 0 and t = 1', () => {
    expect(mix('#000000', '#ffffff', 0)).toBe('#000000')
    expect(mix('#000000', '#ffffff', 1)).toBe('#ffffff')
  })

  it('blends each channel', () => {
    expect(mix('#000000', '#ff8040', 0.5)).toBe('#804020')
  })
})

describe('ramp', () => {
  const stops = ['#000000', '#ff0000', '#ffffff']

  it('hits every stop exactly', () => {
    expect(ramp(stops, 0)).toBe('#000000')
    expect(ramp(stops, 0.5)).toBe('#ff0000')
    expect(ramp(stops, 1)).toBe('#ffffff')
  })

  it('clamps t outside [0, 1]', () => {
    expect(ramp(stops, -1)).toBe('#000000')
    expect(ramp(stops, 2)).toBe('#ffffff')
  })
})
