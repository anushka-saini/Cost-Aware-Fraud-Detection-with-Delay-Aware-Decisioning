import { useQuery } from '@tanstack/react-query'
import { ArrowDown, ArrowUp, Equal, Trophy } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Async, ChartTooltip, Legend, Note, PageHeader } from '@/components/common'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api, type HeldOutSide, type LeaderboardRow, type Test, type V3Summary, type Verdict } from '@/lib/api'
import { cn, fmt, sentence } from '@/lib/utils'
import V2Study from './ModelComparison'

// Colour follows the model, never its rank. V2 is also dashed, so the two
// deployed models can be told apart without colour.
const SERIES: Record<string, { color: string; dash?: string }> = {
  lightgbm: { color: 'var(--series-1)' },
  v2_deployed: { color: 'var(--series-2)', dash: '5 4' },
  catboost: { color: 'var(--series-3)' },
  mlp: { color: 'var(--series-4)' },
}
const CHART_MODELS = Object.keys(SERIES)
const shortLabel = (key: string, label: string) => (key === 'lightgbm' ? 'V3 (LightGBM)' : key === 'v2_deployed' ? 'V2 deployed' : label)

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

const VERDICT: Record<Verdict, { icon: typeof ArrowUp; text: string; className: string }> = {
  'significantly better': { icon: ArrowUp, text: 'Better', className: 'border-allow/30 bg-allow-soft text-allow-ink' },
  'significantly worse': { icon: ArrowDown, text: 'Worse', className: 'border-block/30 bg-block-soft text-block-ink' },
  comparable: { icon: Equal, text: 'Comparable', className: 'bg-surface-2 text-secondary' },
  reference: { icon: Trophy, text: 'Reference', className: 'border-accent/40 text-accent' },
}

function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const { icon: Icon, text, className } = VERDICT[verdict]
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs font-medium', className)}>
      <Icon className="h-3 w-3" aria-hidden="true" />{text}
    </span>
  )
}

const ci = (low: number, high: number, digits = 4) => `[${low.toFixed(digits)}, ${high.toFixed(digits)}]`
const signedInt = (v: number) => `${v >= 0 ? '+' : '−'}${fmt.int(Math.abs(v))}`

function Difference({ test, format }: { test: Test; format: (v: number) => string }) {
  return (
    <span className="tabular">
      {format(test.difference)} <span className="text-muted">({format(test.ci_low)} to {format(test.ci_high)})</span>
    </span>
  )
}

// ---------------------------------------------------------------------------
// V1 -> V2 -> V3
// ---------------------------------------------------------------------------

function History({ data }: { data: V3Summary }) {
  return (
    <Card>
      <CardHeader title="Version history: V1 → V2 → V3" description="Each version on the same held-out days (14–16). V1's figures are not comparable: they were tuned on these days." />
      <CardBody className="grid gap-3 pt-2 md:grid-cols-3">
        {data.history.map((h) => (
          <div key={h.version} className={cn('rounded-lg border p-4', h.version === 'V3' && 'border-accent/50 bg-surface-2')}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-lg font-semibold">{h.version}</span>
              {h.version === 'V3' && <span className="rounded border border-accent/40 px-1.5 text-[10px] font-semibold tracking-wide text-accent uppercase">Deployed</span>}
              {!h.honest && <span className="rounded border border-review/40 bg-review-soft px-1.5 text-[10px] font-semibold tracking-wide text-review-ink uppercase">Not honest</span>}
            </div>
            <div className="text-xs text-muted">{h.model}</div>
            <dl className="mt-3 space-y-1 text-[13px]">
              <div className="flex justify-between gap-2"><dt className="text-secondary">PR-AUC</dt>
                <dd className="tabular font-medium">{h.pr_auc.toFixed(4)} <span className="font-normal text-muted">{ci(h.ci_low, h.ci_high, 3)}</span></dd></div>
              <div className="flex justify-between gap-2"><dt className="text-secondary">Total cost</dt><dd className="tabular font-medium">{fmt.int(h.total_cost)}</dd></div>
              <div className="flex justify-between gap-2"><dt className="text-secondary">Frauds let through</dt><dd className="tabular">{h.fraud_allowed}</dd></div>
              <div className="flex justify-between gap-2"><dt className="text-secondary">Wrongful blocks</dt><dd className="tabular">{h.legit_blocked}</dd></div>
            </dl>
            <p className={cn('mt-3 text-xs', h.honest ? 'text-muted' : 'text-review-ink')}>{h.note}.</p>
          </div>
        ))}
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// The one-time held-out test
// ---------------------------------------------------------------------------

function HeldOut({ data }: { data: V3Summary }) {
  const h = data.held_out
  type Row = { label: string; get: (s: HeldOutSide) => string; diff?: React.ReactNode }
  const rows: Row[] = [
    { label: 'PR-AUC', get: (s) => s.pr_auc.pr_auc.toFixed(4), diff: <Difference test={h.difference.pr_auc} format={(v) => fmt.signed(v)} /> },
    { label: 'ROC-AUC', get: (s) => s.roc_auc.roc_auc.toFixed(4), diff: <Difference test={h.difference.roc_auc} format={(v) => fmt.signed(v, 5)} /> },
    { label: 'Total cost', get: (s) => fmt.int(s.total_cost), diff: <Difference test={h.difference.cost} format={signedInt} /> },
    { label: 'Frauds let through', get: (s) => String(s.fraud_allowed) },
    { label: 'Legitimate reviewed', get: (s) => fmt.pct(s.legit_reviewed_pct) },
    { label: 'Wrongful blocks', get: (s) => String(s.legit_blocked) },
    { label: 'Block recall', get: (s) => fmt.pct(s.block_recall, 1) },
    { label: 'Thresholds (review / block)', get: (s) => `${fmt.threshold(s.t_review)} / ${s.t_block}` },
  ]
  return (
    <Card>
      <CardHeader title={`The one-time held-out test (days ${h.days[0]}–${h.days[h.days.length - 1]})`}
        description={`${fmt.int(h.n_rows)} transactions, ${h.n_fraud} fraud. The rule was written into the script before these days were scored, and the script refuses to run twice.`} />
      <CardBody className="space-y-3 pt-2">
        <Note>{h.decision_rule}</Note>
        <div className={cn('rounded-md border px-3 py-2 text-[13px] font-medium', h.deploy_v3 ? 'border-allow/30 bg-allow-soft text-allow-ink' : 'border-review/40 bg-review-soft text-review-ink')}>
          Result: {h.verdict}
        </div>
        <Table>
          <thead><tr><Th>Measure</Th><Th className="text-right">V3</Th><Th className="text-right">V2 deployed</Th><Th className="text-right">V3 − V2 (95% CI)</Th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <Td className="text-secondary">{r.label}</Td>
                <Td className="tabular text-right font-medium">{r.get(h.v3)}</Td>
                <Td className="tabular text-right">{r.get(h.v2)}</Td>
                <Td className="text-right">{r.diff ?? <span className="text-muted">–</span>}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
        <p className="text-xs text-muted">
          For scale: blocking nothing would have cost {fmt.money(h.baselines.block_nothing)} on these days, and reviewing every transaction {fmt.int(h.baselines.review_everything)}.
          Costs: a missed fraud costs its amount, a wrongful block 5, a review 1.
        </p>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Leaderboard (days 11-13)
// ---------------------------------------------------------------------------

function IntervalBar({ value, low, high, axis }: { value: number; low: number; high: number; axis: [number, number] }) {
  const x = (v: number) => ((v - axis[0]) / (axis[1] - axis[0])) * 100
  if (high < axis[0]) return <span className="text-xs text-muted">◄ below {axis[0].toFixed(2)}</span>
  return (
    <div className="relative h-4 min-w-36" title={`${value.toFixed(4)} ${ci(low, high)}`}>
      <span className="absolute top-2 right-0 left-0 h-px bg-border" />
      <span className="absolute top-[7px] h-0.5 rounded bg-[var(--series-1)]" style={{ left: `${Math.max(x(low), 0)}%`, width: `${x(high) - Math.max(x(low), 0)}%` }} />
      <span className="absolute top-0.5 h-3 w-3 -translate-x-1/2 rounded-full bg-[var(--series-1)] ring-2 ring-surface" style={{ left: `${x(value)}%` }} />
    </div>
  )
}

function Leaderboard({ data }: { data: V3Summary }) {
  const rows = data.validation.leaderboard
  const main = rows.filter((r) => r.pr_auc).sort((a, b) => b.pr_auc!.pr_auc - a.pr_auc!.pr_auc)
  const dayOnly = rows.filter((r) => !r.pr_auc && r.day_13_pr_auc)
  // Axis for the interval bars: from just below the weakest good model to just above the best.
  const lows = main.map((r) => r.pr_auc!.ci_low).filter((v) => v > 0.7)
  const axis: [number, number] = [Math.floor(Math.min(...lows) * 50) / 50, Math.ceil(Math.max(...main.map((r) => r.pr_auc!.ci_high)) * 50) / 50]

  const row = (r: LeaderboardRow, rank: number | null) => {
    const pr = r.pr_auc ?? r.day_13_pr_auc!
    return (
      <tr key={r.key} className={cn(r.key === data.v3_key && 'bg-surface-2')}>
        <Td className="tabular text-muted">{rank ?? '–'}</Td>
        <Td className="font-medium">
          {r.key === data.v3_key ? 'LightGBM (tuned), now V3' : r.label}
          {r.key === data.v3_key && <span className="ml-2 rounded border border-accent/40 px-1 text-[10px] font-semibold tracking-wide text-accent uppercase">Deployed</span>}
          {!r.pr_auc && <div className="text-[11px] font-normal text-muted">judged on day 13 only</div>}
        </Td>
        <Td className="text-secondary capitalize">{r.kind}</Td>
        <Td className="tabular text-right">{pr.pr_auc.toFixed(4)}<div className="text-[11px] text-muted">{ci(pr.ci_low, pr.ci_high, 3)}</div></Td>
        <Td>{r.pr_auc ? <IntervalBar value={pr.pr_auc} low={pr.ci_low} high={pr.ci_high} axis={axis} /> : null}</Td>
        <Td className="tabular text-right">{r.roc_auc ? r.roc_auc.roc_auc.toFixed(4) : '–'}</Td>
        <Td className="tabular text-right text-secondary">{r.seed_mean != null && r.seed_std != null ? `${r.seed_mean.toFixed(4)} ± ${r.seed_std.toFixed(4)}` : '–'}</Td>
        <Td className="tabular text-right">{fmt.int(r.day_13_cost)}</Td>
        <Td><VerdictBadge verdict={r.verdict_vs_v3} /></Td>
        <Td><VerdictBadge verdict={r.verdict_vs_v2} /></Td>
      </tr>
    )
  }

  return (
    <Card>
      <CardHeader title="Leaderboard: every model on days 11–13"
        description="Hyperparameters were tuned on folds inside days 5–10. Nothing here looked at the held-out days." />
      <CardBody className="space-y-2 overflow-x-auto pt-2">
        <Table>
          <thead><tr>
            <Th>#</Th><Th>Model</Th><Th>Kind</Th><Th className="text-right">PR-AUC [95% CI]</Th>
            <Th>Interval ({axis[0].toFixed(2)} to {axis[1].toFixed(2)})</Th><Th className="text-right">ROC-AUC</Th>
            <Th className="text-right">5 seeds, mean ± std</Th><Th className="text-right">Day-13 cost</Th><Th>vs V3</Th><Th>vs V2</Th>
          </tr></thead>
          <tbody>
            {main.map((r, i) => row(r, i + 1))}
            {dayOnly.map((r) => row(r, null))}
          </tbody>
        </Table>
        <p className="text-xs text-muted">
          "Better" and "worse" need a paired bootstrap win that survives the Holm correction ({data.validation.significance_rule}); anything else is "comparable".
          Day-13 cost: calibration and thresholds fitted on days 11–12, cost measured on day 13. {sentence(data.validation.stacking_note)}.
        </p>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

const fprTick = (v: number) => (v >= 1 ? '100%' : v >= 0.01 ? `${v * 100}%` : `${(v * 100).toFixed(v >= 0.001 ? 1 : 3)}%`)

function Curves({ data }: { data: V3Summary }) {
  const c = data.curves!
  const models = CHART_MODELS.filter((k) => c.curves[k])
  const legend = models.map((k) => ({ label: shortLabel(k, c.curves[k].label), color: SERIES[k].color }))
  const pr = c.curves[models[0]].pr.map((p, i) => ({ recall: p.recall, ...Object.fromEntries(models.map((k) => [k, c.curves[k].pr[i].precision])) }))
  const roc = c.curves[models[0]].roc.map((p, i) => ({ fpr: p.fpr, ...Object.fromEntries(models.map((k) => [k, c.curves[k].roc[i].tpr])) }))
  const name = (key: string) => shortLabel(key, c.curves[key].label)

  const lines = models.map((k) => (
    <Line key={k} dataKey={k} name={name(k)} type="monotone" stroke={SERIES[k].color} strokeDasharray={SERIES[k].dash}
      strokeWidth={k === data.v3_key ? 2.5 : 2} dot={false} isAnimationActive={false} />
  ))

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <CardHeader title="Precision against recall" description={`On ${c.curves_on}. ${sentence(c.pr_note)}. Below 50% recall every model is at 100% precision, so the axis starts there.`} />
        <CardBody className="pt-0">
          <Legend items={legend} />
          <div className="mt-2 h-72">
          <ResponsiveContainer>
            <LineChart data={pr.filter((p) => p.recall >= 0.5)} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="recall" type="number" domain={[0.5, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false}
                label={{ value: 'Recall (share of fraud caught)', position: 'insideBottom', offset: -2, fontSize: 11 }} height={36} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} width={44} />
              <Tooltip cursor={{ stroke: 'var(--text-muted)', strokeDasharray: '3 3' }}
                content={<ChartTooltip labelFormatter={(v) => `Recall ${Math.round(Number(v) * 100)}%`} formatter={(v) => `${(v * 100).toFixed(1)}% precision`} />} />
              {lines}
            </LineChart>
          </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Fraud caught against false alarms" description={`ROC curve on ${c.curves_on}, with false alarms on a log scale: ${c.roc_note.replace('true-positive rate at ', '').replace(' spaced on a log scale', '')}.`} />
        <CardBody className="pt-0">
          <Legend items={legend} />
          <div className="mt-2 h-72">
          <ResponsiveContainer>
            <LineChart data={roc} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="fpr" type="number" scale="log" domain={[1e-5, 1]} ticks={[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]} tickFormatter={fprTick}
                tickLine={false} axisLine={false} label={{ value: 'Legitimate transactions flagged (false-positive rate)', position: 'insideBottom', offset: -2, fontSize: 11 }} height={36} />
              <YAxis domain={[0.6, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} width={44} allowDataOverflow />
              <Tooltip cursor={{ stroke: 'var(--text-muted)', strokeDasharray: '3 3' }}
                content={<ChartTooltip labelFormatter={(v) => `${fprTick(Number(v))} of legitimate flagged`} formatter={(v) => `${(v * 100).toFixed(1)}% of fraud caught`} />} />
              {lines}
            </LineChart>
          </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>

      <Calibration data={data} />
    </div>
  )
}

function Calibration({ data }: { data: V3Summary }) {
  const c = data.curves!
  const models = ['lightgbm', 'v2_deployed'].filter((k) => c.calibration[k])
  const points = (k: string) => c.calibration[k].bins.filter((b) => b.observed_rate > 0).map((b) => ({ x: b.mean_predicted, y: b.observed_rate, n: b.n, fraud: b.n_fraud }))
  const v3 = c.calibration[data.v3_key]
  const zeroBin = v3.bins.find((b) => b.observed_rate === 0)

  return (
    <Card className="xl:col-span-2">
      <CardHeader title="Are the probabilities honest?"
        description={`Calibration on ${c.calibration_on}. Points on the diagonal mean "a 3% score really is fraud about 3% of the time".`}
        action={<Legend items={models.map((k) => ({ label: shortLabel(k, c.calibration[k].label), color: SERIES[k].color }))} />} />
      <CardBody className="grid gap-4 pt-0 lg:grid-cols-[1fr_1fr]">
        <div className="h-72">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid />
              <XAxis dataKey="x" type="number" scale="log" domain={[1e-4, 1]} ticks={[1e-4, 1e-3, 1e-2, 1e-1, 1]} tickFormatter={fprTick} tickLine={false} axisLine={false}
                label={{ value: 'Average predicted probability in the bin', position: 'insideBottom', offset: -2, fontSize: 11 }} height={36} allowDataOverflow />
              <YAxis dataKey="y" type="number" scale="log" domain={[1e-4, 1]} ticks={[1e-4, 1e-3, 1e-2, 1e-1, 1]} tickFormatter={fprTick} tickLine={false} axisLine={false} width={52} allowDataOverflow />
              <ReferenceLine segment={[{ x: 1e-4, y: 1e-4 }, { x: 1, y: 1 }]} stroke="var(--text-muted)" strokeDasharray="4 3" />
              <Tooltip content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const p = payload[0].payload as { x: number; y: number; n: number; fraud: number }
                return (
                  <div className="rounded-md border bg-surface px-2.5 py-2 text-xs shadow-md">
                    <div className="font-medium">{payload[0].name}</div>
                    <div className="tabular">Predicted {fmt.probability(p.x)}, observed {fmt.probability(p.y)}</div>
                    <div className="tabular text-secondary">{p.fraud} fraud in {fmt.int(p.n)} transactions</div>
                  </div>
                )
              }} />
              {models.map((k) => (
                <Scatter key={k} name={shortLabel(k, c.calibration[k].label)} data={points(k)} fill={SERIES[k].color} line={{ stroke: SERIES[k].color, strokeDasharray: SERIES[k].dash }}
                  shape={k === 'v2_deployed' ? 'diamond' : 'circle'} isAnimationActive={false} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-2">
          <Table>
            <thead><tr><Th>V3 score bin</Th><Th className="text-right">Transactions</Th><Th className="text-right">Fraud</Th><Th className="text-right">Predicted</Th><Th className="text-right">Observed</Th></tr></thead>
            <tbody>
              {v3.bins.map((b) => (
                <tr key={b.from}>
                  <Td className="tabular text-secondary">{fmt.probability(b.from)} to {b.to === 1 ? '100%' : fmt.probability(b.to)}</Td>
                  <Td className="tabular text-right">{fmt.int(b.n)}</Td>
                  <Td className="tabular text-right">{b.n_fraud}</Td>
                  <Td className="tabular text-right">{fmt.probability(b.mean_predicted)}</Td>
                  <Td className="tabular text-right">{b.observed_rate === 1 ? '100%' : fmt.probability(b.observed_rate)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          <p className="text-xs text-muted">
            Brier score on day 13: V3 {v3.brier.toExponential(3)}{c.calibration.v2_deployed && <>, V2 {c.calibration.v2_deployed.brier.toExponential(3)}</>}.
            {zeroBin && <> The lowest bin ({fmt.int(zeroBin.n)} transactions, no fraud) cannot be drawn on a log scale, so it is only in the table.</>}
          </p>
        </div>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Phase 2 choices
// ---------------------------------------------------------------------------

function Choices({ data }: { data: V3Summary }) {
  const p = data.phase2
  return (
    <Card>
      <CardHeader title="How V3's probabilities and thresholds were chosen" description="Phase 2, on days 11–13 only." />
      <CardBody className="pt-2">
        <dl className="grid gap-x-6 gap-y-3 text-[13px] md:grid-cols-2">
          <div><dt className="text-xs text-muted">Calibration</dt><dd className="font-medium capitalize">{p.calibration}</dd>
            {p.isotonic_rejected_because && <dd className="text-secondary">Isotonic was not used: {p.isotonic_rejected_because}.</dd>}</div>
          <div><dt className="text-xs text-muted">Threshold rule</dt><dd className="text-secondary">{p.threshold_rule}.</dd></div>
          <div><dt className="text-xs text-muted">Thresholds</dt>
            <dd className="tabular font-medium">review at {fmt.threshold(p.point.t_review)} or above, block at {p.point.t_block} or above</dd></div>
          <div><dt className="text-xs text-muted">Distance to the cost cliff</dt>
            <dd className="tabular font-medium">×{p.odds_factor_to_cliff.toFixed(2)} in odds</dd>
            <dd className="text-secondary">How much every score could shrink before the next fraud slips under the review threshold.</dd></div>
          <div><dt className="text-xs text-muted">Cost on days 11–13</dt>
            <dd className="tabular font-medium">{fmt.int(p.point.total_cost)}, {p.point.fraud_allowed} frauds let through, {p.point.legit_blocked} wrongful blocks</dd></div>
          <div><dt className="text-xs text-muted">Baselines on days 11–13</dt>
            <dd className="tabular text-secondary">block nothing {fmt.money(p.baselines.block_nothing)} · review everything {fmt.int(p.baselines.review_everything)}
              {p.baselines.v2_deployed != null && <> · V2 deployed {fmt.int(p.baselines.v2_deployed)}</>}</dd></div>
        </dl>
      </CardBody>
    </Card>
  )
}

function Body({ data }: { data: V3Summary }) {
  return (
    <div className="space-y-4">
      <History data={data} />
      <HeldOut data={data} />
      <Leaderboard data={data} />
      {data.curves && <Curves data={data} />}
      <Choices data={data} />
      <p className="text-xs text-muted">
        Every figure on this page is read from results/metrics.json. Sources: {Object.entries(data.sources).map(([section, source]) => `${section} (${source})`).join('; ')}.
      </p>
    </div>
  )
}

const TABS = [
  { id: 'v3', label: 'Version 3 (current)' },
  { id: 'v2', label: 'V2 study (superseded)' },
] as const

export default function ModelLab() {
  const [params, setParams] = useSearchParams()
  const view = params.get('view') === 'v2' ? 'v2' : 'v3'
  const query = useQuery({ queryKey: ['results-v3'], queryFn: api.v3, staleTime: Infinity })

  return (
    <>
      <PageHeader title="Model lab"
        description="How V3 was chosen: every model under one protocol, the one-time held-out test against V2, and how the project got from V1 to V3." />
      <div className="mb-4 inline-flex rounded-md border bg-surface p-0.5" role="tablist" aria-label="Which results">
        {TABS.map((tab) => (
          <button key={tab.id} role="tab" aria-selected={view === tab.id} onClick={() => setParams(tab.id === 'v3' ? {} : { view: tab.id })}
            className={cn('rounded px-3 py-1 text-[13px]', view === tab.id ? 'bg-surface-2 font-medium' : 'text-secondary hover:text-text')}>
            {tab.label}
          </button>
        ))}
      </div>
      {view === 'v2' ? <V2Study /> : (
        <Async query={query} rows={8}>
          {(data) => data.available ? <Body data={data} />
            : <Note tone="warning">Result sections are missing ({data.missing_sections?.join(', ')}). Run scripts/v3_phase1.py compare, v3_phase2.py and v3_phase3.py.</Note>}
        </Async>
      )}
    </>
  )
}
