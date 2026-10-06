// All requests go through the Vite proxy (/api -> FastAPI on port 8000).
const BASE = '/api'

export const API_START_HINT =
  'python -m uvicorn main:app --host 0.0.0.0 --port 8000 (run from the api/ folder)'

async function request(path, options) {
  let response
  try {
    response = await fetch(BASE + path, options)
  } catch {
    throw new Error('Could not reach the API.')
  }
  if (!response.ok) {
    // The Vite proxy answers 500/502/504 itself when nothing is listening on port 8000.
    const body = await response.text().catch(() => '')
    throw new Error(
      body ? `API returned HTTP ${response.status}: ${body.slice(0, 200)}` : `API returned HTTP ${response.status}.`,
    )
  }
  return response.json()
}

const post = (path, body) =>
  request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const getHealth = () => request('/health')
export const getModelInfo = () => request('/model-info')
export const getValidationMetrics = () => request('/validation-metrics')
export const getDecisionLog = () => request('/decision-log')
export const getRandomDemoTransaction = () => request('/demo-sample/random')
export const scoreTransaction = (txn) => post('/score', txn)
export const explainTransaction = (txn) => post('/explain', txn)
export const confirmDecision = (assessmentId, userProceeded) =>
  post('/confirm-decision', { assessment_id: assessmentId, user_proceeded: userProceeded })
