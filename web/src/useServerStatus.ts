import { useEffect, useState } from 'react'

import { errorMessage, health } from './api'

export type ServerState = 'checking' | 'waking' | 'ready' | 'unreachable'

export interface ServerStatus {
  state: ServerState
  elapsedS: number
  latencyMs?: number
  error?: string
  retry: () => void
}

// A warm server answers /health well under this; a slower answer means a cold start
const WAKING_AFTER_S = 3

/**
 * Calls /health once on page load to wake the Render instance, and tracks how
 * long it takes. The Live API tab shows the result.
 */
export function useServerStatus(): ServerStatus {
  const [attempt, setAttempt] = useState(0)
  const [elapsedS, setElapsedS] = useState(0)
  const [result, setResult] = useState<{ state: 'ready' | 'unreachable'; latencyMs?: number; error?: string } | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const start = performance.now()
    const timer = setInterval(() => setElapsedS(Math.floor((performance.now() - start) / 1000)), 1000)

    health(controller.signal)
      .then(({ ms }) => setResult({ state: 'ready', latencyMs: ms }))
      .catch((err) => {
        if (!controller.signal.aborted) setResult({ state: 'unreachable', error: errorMessage(err) })
      })
      .finally(() => clearInterval(timer))

    return () => {
      controller.abort()
      clearInterval(timer)
    }
  }, [attempt])

  return {
    state: result?.state ?? (elapsedS >= WAKING_AFTER_S ? 'waking' : 'checking'),
    elapsedS,
    latencyMs: result?.latencyMs,
    error: result?.error,
    retry: () => {
      setResult(null)
      setElapsedS(0)
      setAttempt((a) => a + 1)
    },
  }
}
