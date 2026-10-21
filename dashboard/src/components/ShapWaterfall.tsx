import { useState } from 'react'
import type { Explanation } from '@/lib/api'
import { fmt } from '@/lib/utils'
import { Legend } from './common'

const TOWARD = 'var(--block)'      // pushes the score toward fraud
const AWAY = 'var(--series-1)'     // pushes it away
const MAX_ROWS = 9

const featureValue = (value: number | boolean) =>
  typeof value === 'boolean' ? (value ? 'yes' : 'no') : Number.isInteger(value) ? fmt.int(value) : Math.abs(value) < 10 ? value.toFixed(3) : fmt.money(value)

// A waterfall: starting from the model's average output, each feature moves
// the running total right (toward fraud) or left (away), ending at this
// transaction's score. Largest effects first; the rest are grouped.
export function ShapWaterfall({ explanation }: { explanation: Explanation }) {
  const [hover, setHover] = useState<number | null>(null)
  const sorted = [...explanation.contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
  const shown = sorted.slice(0, MAX_ROWS)
  const rest = sorted.slice(MAX_ROWS)
  const restSum = rest.reduce((sum, c) => sum + c.shap_value, 0)
  const rows = [
    ...shown.map((c) => ({ label: c.label, feature: c.feature, value: c.shap_value, detail: `${c.feature} = ${featureValue(c.feature_value)}` })),
    ...(rest.length ? [{ label: `${rest.length} other features`, feature: 'other', value: restSum, detail: 'combined effect' }] : []),
  ]

  let running = explanation.base_value
  const steps = rows.map((row) => {
    const start = running
    running += row.value
    return { ...row, start, end: running }
  })
  const final = running
  const low = Math.min(explanation.base_value, ...steps.map((s) => Math.min(s.start, s.end)))
  const high = Math.max(explanation.base_value, ...steps.map((s) => Math.max(s.start, s.end)))
  const span = high - low || 1
  const x = (v: number) => ((v - low) / span) * 100
  const unit = explanation.units === 'log-odds' ? 'log-odds' : 'probability'

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <Legend items={[{ label: 'Pushes toward fraud', color: TOWARD }, { label: 'Pushes away from fraud', color: AWAY }]} />
        <span className="text-xs text-muted">Contributions in {unit} (SHAP), averaged over the ensemble</span>
      </div>
      <div className="space-y-1" role="table" aria-label="Feature contributions">
        <div className="grid grid-cols-[minmax(0,15rem)_1fr_4.5rem] items-center gap-3 text-xs text-muted">
          <span>Average transaction</span>
          <div className="relative h-4"><span className="absolute top-0 h-4 w-px bg-muted" style={{ left: `${x(explanation.base_value)}%` }} /></div>
          <span className="tabular text-right">{explanation.base_value.toFixed(2)}</span>
        </div>
        {steps.map((step, index) => {
          const left = x(Math.min(step.start, step.end))
          const width = Math.max(Math.abs(x(step.end) - x(step.start)), 0.4)
          const toward = step.value >= 0
          return (
            <div key={step.feature} role="row" onMouseEnter={() => setHover(index)} onMouseLeave={() => setHover(null)}
              className="grid grid-cols-[minmax(0,15rem)_1fr_4.5rem] items-center gap-3 rounded text-[13px] hover:bg-surface-2">
              <span className="truncate" title={step.detail}>
                {step.label.charAt(0).toUpperCase() + step.label.slice(1)}
                {hover === index && <span className="ml-2 font-mono text-[11px] text-muted">{step.detail}</span>}
              </span>
              <div className="relative h-5">
                <span className="absolute top-0 h-5 w-px bg-border" style={{ left: `${x(explanation.base_value)}%` }} />
                <span className="absolute top-1 h-3 rounded-[3px]" style={{ left: `${left}%`, width: `${width}%`, background: toward ? TOWARD : AWAY }} />
              </div>
              <span className="tabular text-right font-medium">{toward ? '+' : '−'}{Math.abs(step.value).toFixed(2)}</span>
            </div>
          )
        })}
        <div className="grid grid-cols-[minmax(0,15rem)_1fr_4.5rem] items-center gap-3 border-t pt-1 text-[13px] font-semibold">
          <span>This transaction</span>
          <div className="relative h-4"><span className="absolute top-0 h-4 w-0.5 bg-text" style={{ left: `${x(final)}%` }} /></div>
          <span className="tabular text-right">{final.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}
