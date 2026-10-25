import type { ParsedInputs } from './inputs'
import type { ShapContribution } from './model/logreg'

export const API_URL = (import.meta.env.VITE_API_URL ?? 'https://alzheimers-ml.onrender.com').replace(/\/$/, '')

export const HEALTH_TIMEOUT_MS = 90_000   // Render free-tier cold starts can take about a minute
export const PREDICT_TIMEOUT_MS = 60_000

export interface HealthResponse {
  status: string
  model: string
}

/** The API's PredictResponse (api/schemas.py). Values are rounded by the server. */
export interface ApiPrediction {
  predicted_class: 'AD' | 'CN'
  probability_ad: number
  probability_cn: number
  shap_contributions: ShapContribution[]
  disclaimer: string
}

export interface Timed<T> {
  data: T
  ms: number
}

export class ApiError extends Error {}

async function request<T>(path: string, init: RequestInit, timeoutMs: number, signal?: AbortSignal): Promise<Timed<T>> {
  const signals = signal ? [signal, AbortSignal.timeout(timeoutMs)] : [AbortSignal.timeout(timeoutMs)]
  const start = performance.now()
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: AbortSignal.any(signals) })
  } catch (err) {
    if (signal?.aborted) throw err
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new ApiError(`No response after ${timeoutMs / 1000} s. The server may still be starting. Try again.`)
    }
    throw new ApiError(`Could not reach the API at ${API_URL}.`)
  }
  if (!response.ok) {
    throw new ApiError(`API error ${response.status}: ${await response.text()}`)
  }
  const data = (await response.json()) as T
  return { data, ms: performance.now() - start }
}

export function health(signal?: AbortSignal): Promise<Timed<HealthResponse>> {
  return request('/health', {}, HEALTH_TIMEOUT_MS, signal)
}

/** The request body /predict expects. */
export function predictBody(inputs: ParsedInputs) {
  return { amyloid_suvrs: inputs.amyloid, tau_suvrs: inputs.tau }
}

export function predictRemote(inputs: ParsedInputs): Promise<Timed<ApiPrediction>> {
  return request(
    '/predict',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(predictBody(inputs)) },
    PREDICT_TIMEOUT_MS,
  )
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
