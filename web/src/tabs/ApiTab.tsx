import { useState } from 'react'

import { API_URL, errorMessage, predictBody, predictRemote, type ApiPrediction, type Timed } from '../api'
import { parseForm, type FormValues, type ParsedInputs } from '../inputs'
import { predict, PROB_TOLERANCE, SHAP_TOLERANCE, type Prediction } from '../model/logreg'
import { PARAMS } from '../model/params'
import type { ServerStatus } from '../useServerStatus'

interface Props {
  values: FormValues
  server: ServerStatus
}

interface Exchange {
  sent: ParsedInputs
  local: Prediction
  localMs: number
  response?: Timed<ApiPrediction>
  error?: string
}

// A warm server answers /health in well under this; slower means it was asleep
const COLD_START_MS = 3000

/** The in-browser prediction for the same inputs, with how long it took. */
function timedLocalPredict(inputs: ParsedInputs): { local: Prediction; localMs: number } {
  const start = performance.now()
  const local = predict(PARAMS, inputs.amyloid, inputs.tau)
  return { local, localMs: performance.now() - start }
}

const formatMs = (ms: number) => (ms < 1 ? 'under 1 ms' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)

function StatusPill({ server }: { server: ServerStatus }) {
  const text = {
    checking: 'Checking the server',
    waking: `Waking up the server, ${server.elapsedS} s`,
    ready: 'Server ready',
    unreachable: 'Server unreachable',
  }[server.state]
  return (
    <div className="status-row">
      <span className={`status-pill ${server.state}`} role="status">
        <span className="status-dot" aria-hidden="true" />
        {text}
      </span>
      {server.state === 'waking' && (
        <span className="muted small">The free-tier server sleeps when idle and takes 30 to 60 s to start.</span>
      )}
      {server.state === 'ready' && server.latencyMs != null && (
        <span className="muted small">
          Health check answered in {formatMs(server.latencyMs)}
          {server.latencyMs > COLD_START_MS ? ' after a cold start.' : '.'}
        </span>
      )}
      {server.state === 'unreachable' && (
        <>
          <span className="muted small">{server.error}</span>
          <button type="button" className="pill-button" onClick={server.retry}>Retry</button>
        </>
      )}
    </div>
  )
}

function TimingChart({ healthMs, requestMs, localMs }: { healthMs?: number; requestMs?: number; localMs?: number }) {
  const rows: { label: string; ms: number; kind: string }[] = []
  if (healthMs != null) rows.push({ label: healthMs > COLD_START_MS ? 'Cold start' : 'Health check', ms: healthMs, kind: healthMs > COLD_START_MS ? 'cold' : 'warm' })
  if (requestMs != null) rows.push({ label: 'Prediction request', ms: requestMs, kind: 'warm' })
  if (localMs != null) rows.push({ label: 'In your browser', ms: localMs, kind: 'local' })
  if (rows.length === 0) return null
  const max = Math.max(...rows.map((r) => r.ms), 1)

  return (
    <div className="timing">
      <h2>Timing</h2>
      {rows.map((r) => (
        <div className="timing-row" key={r.label}>
          <span>{r.label}</span>
          <span className="timing-track" aria-hidden="true">
            <span className={`timing-bar ${r.kind}`} style={{ width: `max(4px, ${(r.ms / max) * 100}%)` }} />
          </span>
          <span className="timing-value">{formatMs(r.ms)}</span>
        </div>
      ))}
      <p className="muted small">Server times are round trips from your browser, including network time.</p>
    </div>
  )
}

function Comparison({ local, response }: { local: Prediction; response: ApiPrediction }) {
  const localShap = new Map(local.shap_contributions.map((c) => [c.roi, c.shap_value]))
  const probDiff = Math.abs(response.probability_ad - local.probability_ad)
  const shapDiff = Math.max(...response.shap_contributions.map((c) => Math.abs(c.shap_value - (localShap.get(c.roi) ?? NaN))))
  const match = response.predicted_class === local.predicted_class && probDiff <= PROB_TOLERANCE && shapDiff <= SHAP_TOLERANCE

  return (
    <>
      <div className="panel-head">
        <h2>{match ? 'Server and browser agree' : 'Server and browser differ'}</h2>
        <span className={`match-badge ${match ? 'match' : 'differs'}`}>{match ? 'Match' : 'Differs'}</span>
      </div>
      <table className="compare-table">
        <thead>
          <tr><th scope="col"><span className="sr-only">Value</span></th><th scope="col" className="num">Server</th><th scope="col" className="num">Browser</th></tr>
        </thead>
        <tbody>
          <tr><th scope="row">Predicted class</th><td className="num">{response.predicted_class}</td><td className="num">{local.predicted_class}</td></tr>
          <tr><th scope="row">Probability of AD</th><td className="num">{response.probability_ad.toFixed(4)}</td><td className="num">{local.probability_ad.toFixed(4)}</td></tr>
          <tr><th scope="row">Largest SHAP difference</th><td className="num" colSpan={2}>{shapDiff.toExponential(1)}</td></tr>
        </tbody>
      </table>
      <p className="muted small">
        The server rounds probabilities to 4 decimals and SHAP values to 6, so differences smaller than that count as a match.
      </p>
    </>
  )
}

export function ApiTab({ values, server }: Props) {
  const [sending, setSending] = useState(false)
  const [exchange, setExchange] = useState<Exchange | null>(null)
  const inputs = parseForm(values)

  async function send() {
    if (!inputs || sending) return
    const base = { sent: inputs, ...timedLocalPredict(inputs) }
    setSending(true)
    setExchange(base)
    try {
      setExchange({ ...base, response: await predictRemote(inputs) })
    } catch (err) {
      setExchange({ ...base, error: errorMessage(err) })
    } finally {
      setSending(false)
    }
  }

  const curl = `curl -X POST ${API_URL}/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(predictBody(inputs ?? { amyloid: {}, tau: {} }))}'`

  return (
    <>
      <header className="page-intro">
        <h2>Run the same prediction on the server</h2>
        <p className="lede">
          The same model runs in two places. The Predict tab runs it in your browser for instant results. It is also
          deployed as an API, the way a larger model would be served. This tab sends your values to that API and checks
          that both give the same answer.
        </p>
      </header>

      <section className="paths" aria-label="Two ways the model runs">
        <div className="panel path">
          <h3>In your browser</h3>
          <p className="path-where">Used by the Predict tab</p>
          <ol className="flow">
            <li>Your 26 scan values</li>
            <li>The model, built into this page</li>
            <li>Prediction and SHAP values, in under a millisecond</li>
          </ol>
          <p className="path-why">
            This works because the model is small: 26 weights and one formula. Nothing to host, and no waiting for a server.
          </p>
        </div>
        <div className="panel path">
          <h3>On the server</h3>
          <p className="path-where">Used by this tab</p>
          <ol className="flow">
            <li>Your 26 scan values, sent over HTTPS</li>
            <li>FastAPI service in a Docker container on Render</li>
            <li>The model and SHAP's LinearExplainer, in Python</li>
            <li>Prediction and SHAP values, returned as JSON</li>
          </ol>
          <p className="path-why">
            This is the path that scales. Larger models, like a CNN on full PET scans, are too big to run in a browser and
            may need a GPU. Serving them through an API keeps their weights private and lets them be updated in one place.
          </p>
        </div>
      </section>

      <div className="two-col">
        <section className="panel">
          <StatusPill server={server} />
          <div className="endpoint">
            <span className="method">POST</span>
            <code>{API_URL}/predict</code>
          </div>
          <div className="send-row">
            <button type="button" className="primary-button" onClick={send} disabled={!inputs || sending}>
              {sending ? 'Waiting for the server' : 'Send request'}
            </button>
            <span className="muted small">
              {inputs ? 'Sends the 26 values from the Predict tab.' : 'Fix the highlighted values on the Predict tab first.'}
            </span>
          </div>
          <TimingChart healthMs={server.latencyMs} requestMs={exchange?.response?.ms} localMs={exchange?.localMs} />
        </section>

        <section className="panel">
          {exchange?.error && <p className="error" role="alert">{exchange.error}</p>}
          {exchange?.response ? (
            <Comparison local={exchange.local} response={exchange.response.data} />
          ) : (
            !exchange?.error && (
              <p className="empty-state">
                {sending ? 'Waiting for the server to answer.' : "Send a request to compare the server's answer with your browser's."}
              </p>
            )
          )}
          {exchange?.response && (
            <>
              <details className="disclosure">
                <summary>Request body</summary>
                <pre>{JSON.stringify(predictBody(exchange.sent), null, 2)}</pre>
              </details>
              <details className="disclosure" open>
                <summary>Response body</summary>
                <pre>{JSON.stringify(exchange.response.data, null, 2)}</pre>
              </details>
            </>
          )}
          <details className="disclosure">
            <summary>Call the API yourself</summary>
            <pre>{curl}</pre>
            <p className="small">
              Interactive docs: <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">{API_URL}/docs</a>
            </p>
          </details>
        </section>
      </div>
    </>
  )
}
