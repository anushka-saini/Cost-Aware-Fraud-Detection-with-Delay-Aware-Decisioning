import { API_START_HINT } from '../api.js'

export function Loading() {
  return <div className="state">Loading…</div>
}

export function ApiError({ error }) {
  return (
    <div className="state">
      <strong>{error.message}</strong>
      Start the FastAPI backend, then reload: <code>{API_START_HINT}</code>
    </div>
  )
}

export function Pending({ what }) {
  return (
    <div className="state">
      <strong>Pending</strong>
      {what} has not been recorded in <code>api/validation_metrics.json</code> yet.
    </div>
  )
}
