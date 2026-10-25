import { useState } from 'react'

import { API_URL, errorMessage, predictBody, predictRemote, type ApiPrediction, type Timed } from '../api'
import { RoiForm } from '../components/RoiForm'
import { parseForm, type FormValues, type Modality, type ParsedInputs } from '../inputs'
import { predict, PROB_TOLERANCE, SHAP_TOLERANCE } from '../model/logreg'
import { PARAMS } from '../model/params'
import type { ServerStatus } from '../useServerStatus'

interface Props {
  values: FormValues
  onChange: (modality: Modality, roi: string, value: string) => void
  onReset: () => void
  server: ServerStatus
}

interface Exchange {
  sent: ParsedInputs
  response?: Timed<ApiPrediction>
  error?: string
}

const formatMs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)

function StatusLine({ server }: { server: ServerStatus }) {
  const labels = {
    checking: 'Checking server…',
    waking: `Waking up · ${server.elapsedS} s`,
    ready: `Ready${server.latencyMs != null ? ` · health check took ${formatMs(server.latencyMs)}` : ''}`,
    unreachable: 'Unreachable',
  }
  return (
    <div className="status-line">
      <span className={`status-pill ${server.state}`} role="status">
        <span className="status-dot" aria-hidden="true" />
        {labels[server.state]}
      </span>
      {server.state === 'waking' && (
        <span className="muted small">
          The free-tier server sleeps when idle. The first request after a sleep takes 30–60 s.
        </span>
      )}
      {server.state === 'unreachable' && (
        <>
          <span className="muted small">{server.error}</span>
          <button type="button" className="secondary small-button" onClick={server.retry}>
            Retry
          </button>
        </>
      )}
    </div>
  )
}

function Comparison({ sent, response }: { sent: ParsedInputs; response: ApiPrediction }) {
  const local = predict(PARAMS, sent.amyloid, sent.tau)
  const localShap = new Map(local.shap_contributions.map((c) => [c.roi, c.shap_value]))
  const probDiff = Math.abs(response.probability_ad - local.probability_ad)
  const shapDiff = Math.max(...response.shap_contributions.map((c) => Math.abs(c.shap_value - (localShap.get(c.roi) ?? NaN))))
  const match = response.predicted_class === local.predicted_class && probDiff <= PROB_TOLERANCE && shapDiff <= SHAP_TOLERANCE

  return (
    <div className="comparison">
      <div className="comparison-header">
        <h3>Server vs in-browser result</h3>
        <span className={`match-badge ${match ? 'match' : 'differs'}`}>{match ? 'Match' : 'Differs'}</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th></th><th>Server</th><th>Browser</th></tr>
          </thead>
          <tbody>
            <tr><th scope="row">Predicted class</th><td>{response.predicted_class}</td><td>{local.predicted_class}</td></tr>
            <tr><th scope="row">P(AD)</th><td>{response.probability_ad.toFixed(4)}</td><td>{local.probability_ad.toFixed(4)}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Largest difference: {probDiff.toExponential(1)} in P(AD), {shapDiff.toExponential(1)} across the 26 SHAP
        values. The server rounds probabilities to 4 decimals and SHAP values to 6, so differences below that count
        as a match.
      </p>
    </div>
  )
}

export function ApiTab({ values, onChange, onReset, server }: Props) {
  const [sending, setSending] = useState(false)
  const [exchange, setExchange] = useState<Exchange | null>(null)
  const inputs = parseForm(values)

  async function send() {
    if (!inputs || sending) return
    setSending(true)
    setExchange({ sent: inputs })
    try {
      setExchange({ sent: inputs, response: await predictRemote(inputs) })
    } catch (err) {
      setExchange({ sent: inputs, error: errorMessage(err) })
    } finally {
      setSending(false)
    }
  }

  const curl = `curl -X POST ${API_URL}/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(predictBody(inputs ?? { amyloid: {}, tau: {} }))}'`

  return (
    <>
      <p className="intro">
        This tab sends your inputs to the deployed inference API (FastAPI on Render), which runs the same model on the
        server and explains it with SHAP's <code>LinearExplainer</code>. It then checks the server's answer against
        the in-browser result.
      </p>

      <StatusLine server={server} />

      <details className="panel">
        <summary>Inputs (shared with the Predict tab)</summary>
        <RoiForm values={values} onChange={onChange} onReset={onReset} />
      </details>

      <div className="send-row">
        <button type="button" onClick={send} disabled={!inputs || sending}>
          {sending ? 'Waiting for server…' : 'Send request'}
        </button>
        <span className="muted small"><code>POST {API_URL}/predict</code></span>
      </div>

      {exchange?.error && <p className="error" role="alert">{exchange.error}</p>}

      {exchange?.response && (
        <>
          <p className="timing">
            Response in <strong>{formatMs(exchange.response.ms)}</strong>
            <span className="muted small"> (round trip from your browser, including network time)</span>
          </p>
          <Comparison sent={exchange.sent} response={exchange.response.data} />
          <details className="panel">
            <summary>Request body</summary>
            <pre>{JSON.stringify(predictBody(exchange.sent), null, 2)}</pre>
          </details>
          <details className="panel" open>
            <summary>Response body</summary>
            <pre>{JSON.stringify(exchange.response.data, null, 2)}</pre>
          </details>
        </>
      )}

      <details className="panel">
        <summary>Call the API yourself</summary>
        <pre>{curl}</pre>
        <p className="small">
          Interactive docs: <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">{API_URL}/docs</a>
        </p>
      </details>
    </>
  )
}
