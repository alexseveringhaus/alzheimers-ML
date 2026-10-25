import type { Suvrs } from './model/logreg'

// The form keeps raw strings so a field can be cleared and retyped.
// Predictions run only when every field parses to a valid SUVR.

export const SUVR_MIN = 0
export const SUVR_MAX = 5

export type FieldValues = Record<string, string>

export interface FormValues {
  amyloid: FieldValues
  tau: FieldValues
}

export interface ParsedInputs {
  amyloid: Suvrs
  tau: Suvrs
}

export type Modality = keyof FormValues

export function toFormValues(inputs: ParsedInputs): FormValues {
  const toFields = (suvrs: Suvrs) => Object.fromEntries(Object.entries(suvrs).map(([roi, v]) => [roi, String(v)]))
  return { amyloid: toFields(inputs.amyloid), tau: toFields(inputs.tau) }
}

export function isValidSuvr(text: string): boolean {
  if (text.trim() === '') return false
  const value = Number(text)
  return Number.isFinite(value) && value >= SUVR_MIN && value <= SUVR_MAX
}

/** Returns numbers for every field, or null if any field is invalid. */
export function parseForm(values: FormValues): ParsedInputs | null {
  const all = [...Object.values(values.amyloid), ...Object.values(values.tau)]
  if (!all.every(isValidSuvr)) return null
  const toSuvrs = (fields: FieldValues) => Object.fromEntries(Object.entries(fields).map(([roi, t]) => [roi, Number(t)]))
  return { amyloid: toSuvrs(values.amyloid), tau: toSuvrs(values.tau) }
}
