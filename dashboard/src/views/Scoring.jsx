import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  API_START_HINT,
  confirmDecision,
  explainTransaction,
  getRandomDemoTransaction,
  scoreTransaction,
} from '../api.js'
import { amount as formatAmount, probability, MODEL_NAMES } from '../format.js'
import { DataTable, Legend, Segmented } from '../components/ChartCard.jsx'

const TYPES = ['CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT', 'TRANSFER']
const SYSTEM_FIELDS = ['total_transactions', 'total_transaction_amount', 'avg_transaction_amount']

// The same four held-out transactions used as presets in the Streamlit app (app.py).
const PRESETS = [
  {
    name: 'Legitimate',
    actualFraud: 0,
    values: {
      amount: 76431.17, type: 'CASH_OUT', origin_balance_error: -76431.17, destination_balance_error: 0,
      destination_balance_is_zero: false, destination_is_first_transaction: false,
      destination_transactions_last_24h: 1, destination_transactions_last_7d: 11,
      destination_avg_previous_amount: 391891.78, destination_amount_deviation: -315460.61,
      total_transactions: 33528, total_transaction_amount: 4693535568.26, avg_transaction_amount: 139988.53,
    },
  },
  {
    name: 'Fraud, low amount',
    actualFraud: 1,
    values: {
      amount: 48266.8, type: 'TRANSFER', origin_balance_error: 0, destination_balance_error: 48266.8,
      destination_balance_is_zero: true, destination_is_first_transaction: true,
      destination_transactions_last_24h: 0, destination_transactions_last_7d: 0,
      destination_avg_previous_amount: 0, destination_amount_deviation: 48266.8,
      total_transactions: 40218, total_transaction_amount: 6596385476.64, avg_transaction_amount: 164015.75,
    },
  },
  {
    name: 'Fraud, mid amount',
    actualFraud: 1,
    values: {
      amount: 500003.56, type: 'CASH_OUT', origin_balance_error: 0, destination_balance_error: 0,
      destination_balance_is_zero: false, destination_is_first_transaction: false,
      destination_transactions_last_24h: 0, destination_transactions_last_7d: 2,
      destination_avg_previous_amount: 47108.3, destination_amount_deviation: 452895.27,
      total_transactions: 10, total_transaction_amount: 23250735.88, avg_transaction_amount: 2325073.588,
    },
  },
  {
    name: 'Fraud, high amount',
    actualFraud: 1,
    values: {
      amount: 4129482.96, type: 'CASH_OUT', origin_balance_error: 0, destination_balance_error: 0,
      destination_balance_is_zero: false, destination_is_first_transaction: false,
      destination_transactions_last_24h: 1, destination_transactions_last_7d: 9,
      destination_avg_previous_amount: 222913.55, destination_amount_deviation: 3906569.41,
      total_transactions: 26927, total_transaction_amount: 5172213957.19, avg_transaction_amount: 192082.81,
    },
  },
]

const DEFAULTS = {
  amount: 5000, type: 'CASH_OUT', origin_balance_error: 0, destination_balance_error: 0,
  destination_balance_is_zero: false, destination_is_first_transaction: false,
  destination_transactions_last_24h: 0, destination_transactions_last_7d: 0,
  destination_avg_previous_amount: 0, destination_amount_deviation: 0,
  // Left unset for manual entries so the API applies its own median defaults.
  total_transactions: null, total_transaction_amount: null, avg_transaction_amount: null,
}

const NUMBER_FIELDS = [
  { key: 'amount', label: 'Amount', min: 0 },
  { key: 'origin_balance_error', label: 'Origin balance error', hint: 'Expected minus actual sender balance after the transaction' },
  { key: 'destination_balance_error', label: 'Destination balance error', hint: 'Expected minus actual recipient balance after the transaction' },
  { key: 'destination_amount_deviation', label: 'Destination amount deviation', hint: "Amount minus the recipient's average previous amount" },
  { key: 'destination_avg_previous_amount', label: 'Destination average previous amount', min: 0 },
  { key: 'destination_transactions_last_24h', label: 'Destination transactions, last 24h', min: 0 },
  { key: 'destination_transactions_last_7d', label: 'Destination transactions, last 7d', min: 0 },
]

function buildPayload(form) {
  const payload = {
    amount: Number(form.amount),
    origin_balance_error: Number(form.origin_balance_error),
    destination_balance_error: Number(form.destination_balance_error),
    destination_balance_is_zero: form.destination_balance_is_zero ? 1 : 0,
    destination_transactions_last_24h: Number(form.destination_transactions_last_24h),
    destination_transactions_last_7d: Number(form.destination_transactions_last_7d),
    destination_avg_previous_amount: Number(form.destination_avg_previous_amount),
    destination_amount_deviation: Number(form.destination_amount_deviation),
    destination_is_first_transaction: form.destination_is_first_transaction ? 1 : 0,
  }
  for (const t of TYPES) payload[`type_${t}`] = form.type === t
  for (const f of SYSTEM_FIELDS) if (form[f] != null) payload[f] = form[f]
  return payload
}

function formFromTransaction(txn) {
  const type = TYPES.find((t) => txn[`type_${t}`]) || 'CASH_OUT'
  const form = { type }
  for (const key of Object.keys(DEFAULTS)) {
    if (key === 'type') continue
    form[key] = typeof DEFAULTS[key] === 'boolean' ? Boolean(txn[key]) : txn[key]
  }
  return form
}

const TIER_TEXT = {
  ALLOW: 'Below the review threshold. The transaction proceeds.',
  REVIEW: 'At or above the review threshold. Held for a decision.',
  BLOCK: 'At or above the block threshold. Held for a decision.',
}

// ---------------------------------------------------------------------------

function ProbabilityMeter({ prob, tReview, tBlock }) {
  const zones = [
    { tier: 'ALLOW', width: tReview, color: 'var(--good)' },
    { tier: 'REVIEW', width: tBlock - tReview, color: 'var(--warning)' },
    { tier: 'BLOCK', width: 1 - tBlock, color: 'var(--critical)' },
  ]
  const at = (v) => ({ left: `${v * 100}%` })
  return (
    <div className="meter" role="img" aria-label={`Fraud probability ${probability(prob)}; review threshold ${tReview}, block threshold ${tBlock}`}>
      <div className="meter-track">
        {zones.map((z) => (
          <div key={z.tier} className="zone" style={{ flex: `${z.width} 0 0`, background: z.color }} />
        ))}
        <div className="meter-marker" style={at(prob)} />
      </div>
      <div className="meter-labels">
        <span className="start" style={at(0)}>0</span>
        <span style={at(tReview)}>review ≥ {tReview}</span>
        <span style={at(tBlock)}>block ≥ {tBlock}</span>
        <span className="end" style={at(1)}>1</span>
      </div>
    </div>
  )
}

function ShapTooltip({ active, payload, format }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="tooltip">
      <div className="tt-title">{d.feature}</div>
      <div className="tt-row">
        <i className="swatch" style={{ background: d.shap_value >= 0 ? 'var(--diverge-high)' : 'var(--diverge-low)' }} />
        {d.shap_value >= 0 ? 'Raises the fraud score' : 'Lowers the fraud score'}
        <b>{format(d.shap_value)}</b>
      </div>
      <div className="tt-note">
        Feature value: {typeof d.feature_value === 'number' ? formatAmount(d.feature_value) : String(d.feature_value)}
      </div>
      <div className="tt-note">Refers to {d.label}.</div>
    </div>
  )
}

function ShapChart({ explanation }) {
  const [view, setView] = useState('Chart')
  const inProbability = explanation.units === 'probability'
  const format = (v) =>
    inProbability ? `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(2)} pp` : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`
  const top = explanation.contributions.slice(0, 8)
  const rest = explanation.contributions.slice(8).reduce((sum, c) => sum + c.shap_value, 0)
  // Symmetric axis around zero with a round half-step, so the ticks are evenly spaced.
  const largest = Math.max(...top.map((c) => Math.abs(c.shap_value))) || 1
  const magnitude = 10 ** Math.floor(Math.log10(largest / 2))
  const half = [1, 1.5, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s * 2 >= largest)
  const extent = half * 2

  return (
    <>
      <div className="card-head" style={{ marginTop: 22 }}>
        <div className="titles">
          <h3>Top contributing factors (SHAP)</h3>
          <p>
            {inProbability
              ? `Each bar is the feature's contribution to this transaction's fraud probability, in percentage points, relative to the model's base value of ${probability(explanation.base_value)}.`
              : 'Each bar is the feature’s contribution to this transaction’s model score, in log-odds.'}
          </p>
        </div>
        <div className="card-controls">
          <Segmented options={['Chart', 'Table']} value={view} onChange={setView} label="View" />
        </div>
      </div>
      {view === 'Table' ? (
        <DataTable
          columns={[
            { key: 'feature', label: 'Feature' },
            { key: 'feature_value', label: 'Value', num: true, format: (v) => (typeof v === 'number' ? formatAmount(v) : String(v)) },
            { key: 'shap_value', label: 'SHAP contribution', num: true, format },
          ]}
          rows={explanation.contributions}
        />
      ) : (
        <>
          <Legend
            items={[
              { label: 'Raises the fraud score', color: 'var(--diverge-high)' },
              { label: 'Lowers the fraud score', color: 'var(--diverge-low)' },
            ]}
          />
          <ResponsiveContainer width="100%" height={top.length * 30 + 40}>
            <BarChart layout="vertical" data={top} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid horizontal={false} stroke="var(--grid)" />
              <XAxis
                type="number"
                domain={[-extent, extent]}
                ticks={[-extent, -half, 0, half, extent]}
                tick={{ fill: 'var(--muted)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--axis)' }}
                tickLine={false}
                tickFormatter={(v) => (inProbability ? `${Number((v * 100).toFixed(2))} pp` : String(Number(v.toFixed(3))))}
              />
              <YAxis type="category" dataKey="feature" width={236} interval={0} tick={{ fill: 'var(--text-2)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ShapTooltip format={format} />} />
              <ReferenceLine x={0} stroke="var(--axis)" />
              <Bar dataKey="shap_value" barSize={14} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {top.map((c) => (
                  <Cell key={c.feature} fill={c.shap_value >= 0 ? 'var(--diverge-high)' : 'var(--diverge-low)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {explanation.contributions.length > top.length && (
            <p className="note">
              Showing the {top.length} largest of {explanation.contributions.length} features. The remaining{' '}
              {explanation.contributions.length - top.length} sum to {format(rest)}.
            </p>
          )}
        </>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------

export default function Scoring({ modelInfo }) {
  const [form, setForm] = useState(DEFAULTS)
  const [groundTruth, setGroundTruth] = useState(null) // null = manual entry, else 0 / 1
  const [origin, setOrigin] = useState('Manual entry')
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [decision, setDecision] = useState(null) // null | 'proceeded' | 'cancelled' | 'failed'
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const clearResult = () => {
    setResult(null)
    setExplanation(null)
    setDecision(null)
    setError(null)
  }

  const load = (values, truth, label) => {
    setForm({ ...DEFAULTS, ...values })
    setGroundTruth(truth)
    setOrigin(label)
    clearResult()
  }

  const loadRandom = async () => {
    try {
      const sample = await getRandomDemoTransaction()
      if (sample.error) throw new Error(sample.error)
      load(formFromTransaction(sample.transaction), sample.actual_fraud, 'Random held-out transaction')
    } catch (e) {
      setError(e.message)
    }
  }

  const update = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }))
    // An edited transaction is no longer the held-out row its label belonged to.
    setGroundTruth(null)
    setOrigin('Manual entry')
  }

  const submit = async (event) => {
    event.preventDefault()
    clearResult()
    setBusy(true)
    const payload = buildPayload(form)
    try {
      setResult(await scoreTransaction(payload))
      setExplanation(await explainTransaction(payload).catch(() => null))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const decide = async (proceeded) => {
    try {
      const response = await confirmDecision(result.assessment_id, proceeded)
      setDecision(response.status === 'logged' ? (proceeded ? 'proceeded' : 'cancelled') : 'failed')
    } catch {
      setDecision('failed')
    }
  }

  const modelName = modelInfo ? MODEL_NAMES[modelInfo.model_class] || modelInfo.model_class : null
  const flagged = result && result.tier !== 'ALLOW'
  const matches = result && groundTruth != null && (groundTruth === 1) === flagged

  return (
    <>
      <div className="page-head">
        <h2>Live Scoring</h2>
        <p>
          Scores one transaction through the <code>/score</code> endpoint
          {modelName && ` (${modelName}, review ≥ ${modelInfo.thresholds.t_review}, block ≥ ${modelInfo.thresholds.t_block})`}.
          Per-feature SHAP contributions come from <code>/explain</code>.
        </p>
      </div>

      <div className="grid scoring">
        <section className="card">
          <div className="card-head">
            <div className="titles">
              <h3>Transaction</h3>
              <p>Presets and the random draw are real held-out transactions with known labels.</p>
            </div>
          </div>

          <div className="presets">
            {PRESETS.map((p) => (
              <button key={p.name} type="button" className="btn" onClick={() => load(p.values, p.actualFraud, `Preset: ${p.name}`)}>
                {p.name}
              </button>
            ))}
            <button type="button" className="btn" onClick={loadRandom}>
              Random held-out transaction
            </button>
          </div>

          <div className="truth">
            <b>{origin}.</b>{' '}
            {groundTruth == null
              ? 'No ground-truth label.'
              : `Ground truth: ${groundTruth === 1 ? 'fraud' : 'legitimate'} (not sent to the model).`}
            {form.total_transactions != null &&
              ` System-level context from the source row: ${formatAmount(form.total_transactions)} transactions in the hour.`}
          </div>

          <form onSubmit={submit}>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="type">Transaction type</label>
                <select id="type" value={form.type} onChange={(e) => update('type', e.target.value)}>
                  {TYPES.map((t) => (
                    <option key={t}>{t}</option>
                  ))}
                </select>
              </div>
              {NUMBER_FIELDS.map((f) => (
                <div className="field" key={f.key}>
                  <label htmlFor={f.key}>{f.label}</label>
                  <input
                    id={f.key}
                    type="number"
                    step="any"
                    min={f.min}
                    required
                    value={form[f.key]}
                    onChange={(e) => update(f.key, e.target.value)}
                  />
                  {f.hint && <div className="hint">{f.hint}</div>}
                </div>
              ))}
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.destination_balance_is_zero}
                  onChange={(e) => update('destination_balance_is_zero', e.target.checked)}
                />
                Destination balance is zero
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.destination_is_first_transaction}
                  onChange={(e) => update('destination_is_first_transaction', e.target.checked)}
                />
                First transaction to this destination
              </label>
            </div>
            <div className="form-foot">
              <button className="btn primary" type="submit" disabled={busy}>
                {busy ? 'Scoring…' : 'Score transaction'}
              </button>
              <button type="button" className="btn" onClick={() => load({}, null, 'Manual entry')}>
                Reset
              </button>
            </div>
          </form>
        </section>

        <section className="card">
          <div className="card-head">
            <div className="titles">
              <h3>Assessment</h3>
            </div>
          </div>

          {error && (
            <div className="error">
              {error} If the backend is not running, start it with: {API_START_HINT}
            </div>
          )}

          {!result && !error && <div className="state">Score a transaction to see the result.</div>}

          {result && (
            <>
              <div className={`tier-banner ${result.tier}`}>
                <div>
                  <div className="tier">{result.tier}</div>
                  <div className="tier-sub">{TIER_TEXT[result.tier]}</div>
                </div>
                <div className="prob">
                  <div className="value">{probability(result.fraud_probability)}</div>
                  <div className="label">Fraud probability</div>
                </div>
              </div>

              <ProbabilityMeter
                prob={result.fraud_probability}
                tReview={result.thresholds_used.t_review}
                tBlock={result.thresholds_used.t_block}
              />

              {groundTruth != null && (
                <div className={`match ${matches ? 'ok' : 'miss'}`}>
                  {matches ? 'Matches ground truth: ' : 'Does not match ground truth: '}
                  the transaction is {groundTruth === 1 ? 'fraud' : 'legitimate'} and was{' '}
                  {flagged ? `flagged (${result.tier})` : 'allowed'}.
                </div>
              )}

              {flagged && decision == null && (
                <div className="decision">
                  <p>This transaction has not gone through. The choice below is written to the decision log.</p>
                  <div className="actions">
                    <button type="button" className="btn" onClick={() => decide(false)}>
                      Cancel transaction
                    </button>
                    <button type="button" className="btn" onClick={() => decide(true)}>
                      Proceed anyway
                    </button>
                  </div>
                </div>
              )}
              {decision === 'proceeded' && <div className="match">Proceeded anyway. Decision logged.</div>}
              {decision === 'cancelled' && <div className="match">Transaction cancelled. Decision logged.</div>}
              {decision === 'failed' && <div className="error">The decision could not be logged.</div>}

              {explanation ? (
                <ShapChart explanation={explanation} />
              ) : (
                <p className="note">SHAP contributions are unavailable for this request.</p>
              )}

              {result.reasons.length > 0 && (
                <>
                  <p className="note">Reasons returned by /score (top factors raising the score):</p>
                  <ul className="reasons">
                    {result.reasons.map((r) => (
                      <li key={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</li>
                    ))}
                  </ul>
                </>
              )}

              <details>
                <summary>Raw /score response</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </>
          )}
        </section>
      </div>
    </>
  )
}
