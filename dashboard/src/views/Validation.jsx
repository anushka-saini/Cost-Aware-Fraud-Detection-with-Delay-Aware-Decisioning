import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getValidationMetrics } from '../api.js'
import { fixed, pct, MODEL_NAMES } from '../format.js'
import useFetch from '../useFetch.js'
import ChartCard, { DataTable, Legend, Segmented } from '../components/ChartCard.jsx'
import ChartTooltip from '../components/ChartTooltip.jsx'
import { ApiError, Loading, Pending } from '../components/States.jsx'

const TICK = { fill: 'var(--muted)', fontSize: 12 }
const AXIS_LINE = { stroke: 'var(--axis)' }
const LABEL = { fill: 'var(--text-2)', fontSize: 12 }
const CURSOR = { fill: 'var(--surface-2)' }

const hasData = (block) => block && (block.status === 'available' || block.status === 'ongoing')

const PERIOD_COLORS = ['var(--period-1)', 'var(--period-2)', 'var(--period-3)']

// ---------------------------------------------------------------------------
// PR-AUC across the four bug-fix stages
// ---------------------------------------------------------------------------
function StageLabel({ x, y, index, stages }) {
  // A local minimum gets its label below the point so the line does not run through it.
  const v = stages[index].pr_auc
  const dip = index > 0 && index < stages.length - 1 && v < stages[index - 1].pr_auc && v < stages[index + 1].pr_auc
  return (
    <text x={x} y={y + (dip ? 24 : -14)} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text)">
      {fixed(4)(v)}
    </text>
  )
}

function Progression({ block }) {
  if (!hasData(block)) return <Pending what="The PR-AUC progression" />
  const first = block.stages[0]
  const last = block.stages[block.stages.length - 1]
  return (
    <ChartCard
      title="PR-AUC across four data-integrity fixes"
      subtitle={`${block.model}, evaluated on ${block.split}`}
      source={block.source}
      note={`The first figure (${fixed(4)(first.pr_auc)}) was produced by leaked and misaligned features. ${fixed(4)(last.pr_auc)} is the figure after all fixes.`}
      table={{
        columns: [
          { key: 'stage', label: 'Stage' },
          { key: 'pr_auc', label: 'PR-AUC', num: true, format: fixed(4) },
          { key: 'state', label: 'State of the data', wrap: true },
        ],
        rows: block.stages,
      }}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={block.stages} margin={{ top: 24, right: 24, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="stage" tick={{ ...TICK, fontSize: 11 }} axisLine={AXIS_LINE} tickLine={false} interval={0} padding={{ left: 64, right: 64 }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={TICK} axisLine={false} tickLine={false} width={44} tickFormatter={fixed(2)} />
          <Tooltip
            cursor={{ stroke: 'var(--axis)' }}
            content={<ChartTooltip format={fixed(4)} note={(d) => d.state} />}
          />
          <Line
            dataKey="pr_auc"
            name="PR-AUC"
            stroke="var(--series-1)"
            strokeWidth={2}
            isAnimationActive={false}
            dot={{ r: 5, fill: 'var(--series-1)', stroke: 'var(--surface)', strokeWidth: 2 }}
            activeDot={{ r: 6, stroke: 'var(--surface)', strokeWidth: 2 }}
          >
            <LabelList dataKey="pr_auc" content={<StageLabel stages={block.stages} />} />
          </Line>
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Drift: PSI per feature across the three evaluation periods
// ---------------------------------------------------------------------------
function FeatureTick({ x, y, payload, highlighted }) {
  const on = highlighted.includes(payload.value)
  return (
    <text x={x} y={y} dy={4} textAnchor="end" fontSize={12} fontWeight={on ? 700 : 400} fill={on ? 'var(--text)' : 'var(--muted)'}>
      {payload.value}
    </text>
  )
}

function PsiChart({ rows, block, domainMax, ticks, labelModerate = true }) {
  const { periods, thresholds, highlighted_features: highlighted } = block
  return (
    <ResponsiveContainer width="100%" height={rows.length * 46 + 44}>
      <BarChart layout="vertical" data={rows} barGap={2} barCategoryGap="20%" margin={{ top: 18, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        <XAxis type="number" domain={[0, domainMax]} ticks={ticks} tick={TICK} axisLine={AXIS_LINE} tickLine={false} />
        <YAxis type="category" dataKey="feature" width={236} axisLine={false} tickLine={false} interval={0} tick={<FeatureTick highlighted={highlighted} />} />
        <Tooltip cursor={CURSOR} content={<ChartTooltip format={fixed(4)} />} />
        <ReferenceLine x={thresholds.moderate} stroke="var(--muted)" strokeDasharray="4 3" label={labelModerate ? { value: String(thresholds.moderate), position: 'top', ...TICK } : undefined} />
        <ReferenceLine x={thresholds.large} stroke="var(--muted)" strokeDasharray="4 3" label={{ value: String(thresholds.large), position: 'top', ...TICK }} />
        {periods.map((p, i) => (
          <Bar key={p} dataKey={p} name={`${p} (days ${block.period_days[p]})`} fill={PERIOD_COLORS[i]} barSize={8} radius={[0, 4, 4, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

function Drift({ block, periodPrAuc }) {
  if (!hasData(block)) return <Pending what="The drift analysis" />
  const { periods, thresholds, highlighted_features: highlighted } = block
  const mark = (rows) => rows.map((r) => ({ ...r, _highlight: highlighted.includes(r.feature) }))
  const columns = [
    { key: 'feature', label: 'Feature' },
    ...periods.map((p) => ({ key: p, label: `PSI, ${p}`, num: true, format: fixed(4) })),
  ]
  const legend = periods.map((p, i) => ({ label: `${p} (days ${block.period_days[p]})`, color: PERIOD_COLORS[i] }))
  const stable = hasData(periodPrAuc)
    ? `LightGBM PR-AUC over the same periods: ${periodPrAuc.periods.map((p) => `${p.period} ${fixed(4)(p.pr_auc)}`).join(', ')} (95% CIs overlap). `
    : ''
  const reference = `PSI against ${block.reference}. Dashed lines mark ${thresholds.moderate} (moderate shift) and ${thresholds.large} (large shift).`

  return (
    <div className="grid cols-2">
      <ChartCard
        title="Feature drift: transaction-level features"
        subtitle={`${reference} Bold: the two destination-velocity features identified as drifting.`}
        source={block.source}
        note={`${stable}Score PSI: ${periods.map((p) => `${p} ${fixed(4)(block.score_psi[p])}`).join(', ')}.`}
        table={{ columns, rows: mark(block.transaction_level) }}
      >
        <Legend items={legend} />
        <PsiChart rows={block.transaction_level} block={block} domainMax={0.6} ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]} />
      </ChartCard>
      <ChartCard
        title="Feature drift: system-level hourly aggregates"
        subtitle={`${reference} Note the wider scale than the chart on the left.`}
        source={block.source}
        note="These three features are full-hour aggregates merged onto every transaction in that hour. They carry the temporal-leakage caveat disclosed in the report, and their PSI reflects hour-level volume differences between periods."
        table={{ columns, rows: block.system_level }}
      >
        <Legend items={legend} />
        <PsiChart rows={block.system_level} block={block} domainMax={4} ticks={[0, 1, 2, 3, 4]} labelModerate={false} />
      </ChartCard>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Recall by fraud-amount quartile, at the review and block thresholds
// ---------------------------------------------------------------------------
function Recall({ block, deployedModel }) {
  const names = hasData(block) ? block.models.map((m) => m.model) : []
  const [selected, setSelected] = useState(null)
  if (!hasData(block)) return <Pending what="Amount-stratified recall" />
  const current = block.models.find((m) => m.model === (selected || names[0]))
  const deployed = deployedModel && current.model === deployedModel
  return (
    <ChartCard
      title="Recall by fraud-amount quartile"
      subtitle={`${current.model} on ${block.split} · review threshold ${current.t_review}, block threshold ${current.t_block}${deployed ? ' · model currently served by the API' : ''}`}
      controls={<Segmented options={names} value={current.model} onChange={setSelected} label="Model" />}
      source={current.source}
      note={`${block.quartile_basis}. Amount ranges: ${current.quartiles.map((q) => `${q.quartile} ${q.range}`).join(' · ')}.`}
      table={{
        columns: [
          { key: 'quartile', label: 'Quartile' },
          { key: 'range', label: 'Amount range' },
          { key: 'n_fraud', label: 'Fraud cases', num: true },
          { key: 'recall_at_review', label: 'Recall at review', num: true, format: pct(1) },
          { key: 'recall_at_block', label: 'Recall at block', num: true, format: pct(1) },
        ],
        rows: current.quartiles,
      }}
    >
      <Legend
        items={[
          { label: `Recall at review (≥ ${current.t_review})`, color: 'var(--series-1)' },
          { label: `Recall at block (≥ ${current.t_block})`, color: 'var(--series-2)' },
        ]}
      />
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={current.quartiles} barGap={18} margin={{ top: 20, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="quartile" tick={TICK} axisLine={AXIS_LINE} tickLine={false} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={TICK} axisLine={false} tickLine={false} width={44} tickFormatter={pct(0)} />
          <Tooltip
            cursor={CURSOR}
            content={<ChartTooltip format={pct(1)} title={(d) => `${d.quartile}: ${d.range}`} note={(d) => `${d.n_fraud} fraud cases`} />}
          />
          <Bar dataKey="recall_at_review" name="Recall at review" fill="var(--series-1)" barSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            <LabelList dataKey="recall_at_review" position="top" formatter={pct(1)} {...LABEL} />
          </Bar>
          <Bar dataKey="recall_at_block" name="Recall at block" fill="var(--series-2)" barSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            <LabelList dataKey="recall_at_block" position="top" formatter={pct(1)} {...LABEL} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// LightGBM vs Random Forest, PR-AUC with bootstrap confidence intervals
// ---------------------------------------------------------------------------
function CiLabel({ x, y, width, height, index, rows }) {
  const row = rows[index]
  // Bars start at zero, so the bar width gives the pixel scale of the axis.
  const ciHighX = x + (width / row.pr_auc) * row.ci_high
  return (
    <text x={ciHighX + 10} y={y + height / 2} dy={4} fontSize={12}>
      <tspan fill="var(--text)" fontWeight={600}>{fixed(4)(row.pr_auc)}</tspan>
      <tspan fill="var(--text-2)">{`  [${fixed(4)(row.ci_low)}, ${fixed(4)(row.ci_high)}]`}</tspan>
    </text>
  )
}

const CI_COLUMNS = [
  { key: 'model', label: 'Model' },
  { key: 'config', label: 'Configuration', wrap: true },
  { key: 'pr_auc', label: 'PR-AUC', num: true, format: (v) => (v == null ? 'pending' : fixed(4)(v)) },
  { key: 'ci_low', label: '95% CI low', num: true, format: fixed(4) },
  { key: 'ci_high', label: '95% CI high', num: true, format: fixed(4) },
]

// Horizontal PR-AUC bars with bootstrap CI whiskers. Models without a recorded value are left out.
function CiBarChart({ models, labelWidth = 110 }) {
  const rows = models
    .filter((m) => m.pr_auc != null)
    .map((m) => ({ ...m, err: [m.pr_auc - m.ci_low, m.ci_high - m.pr_auc] }))
  return (
    <ResponsiveContainer width="100%" height={rows.length * 62 + 56}>
      <BarChart layout="vertical" data={rows} barCategoryGap="30%" margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        {/* Axis runs past 1.0 only to leave room for the value labels. */}
        <XAxis type="number" domain={[0, 1.4]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={TICK} axisLine={AXIS_LINE} tickLine={false} tickFormatter={fixed(2)} />
        <YAxis type="category" dataKey="model" width={labelWidth} tick={{ ...TICK, fill: 'var(--text-2)' }} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={CURSOR}
          content={<ChartTooltip format={fixed(4)} note={(d) => `95% CI [${fixed(4)(d.ci_low)}, ${fixed(4)(d.ci_high)}] · ${d.config}`} />}
        />
        <Bar dataKey="pr_auc" name="PR-AUC" fill="var(--series-1)" barSize={24} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          <ErrorBar dataKey="err" direction="x" width={8} strokeWidth={1.5} stroke="var(--text)" />
          <LabelList content={<CiLabel rows={rows} />} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function ModelComparison({ block }) {
  if (!hasData(block)) return <Pending what="The model comparison" />
  const pending = block.models.filter((m) => m.pr_auc == null).map((m) => m.model)
  return (
    <ChartCard
      title="Model comparison on PaySim: PR-AUC"
      badge={block.status === 'ongoing' ? <span className="badge ongoing">Ongoing investigation</span> : null}
      subtitle={`${block.split} · ${block.ci_method}`}
      source={block.source}
      note={`This comparison is not finalized. The figures are the results recorded so far and may change as the investigation continues.${pending.length ? ` Pending, not yet recorded: ${pending.join(', ')}.` : ''}`}
      table={{ columns: CI_COLUMNS, rows: block.models }}
    >
      <CiBarChart models={block.models} />
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Each model at its own cost-derived thresholds (notebook 07)
// ---------------------------------------------------------------------------
function OperatingPoints({ block }) {
  if (!hasData(block)) return <Pending what="The MLP comparison (notebooks/07_mlp_comparison.ipynb)" />
  const sv = block.seed_variance
  const bs = block.blind_spot
  return (
    <section className="card">
      <div className="card-head">
        <div className="titles">
          <h3>Each model at its own cost-derived thresholds</h3>
          <p>{block.split}. {block.cost_basis}</p>
        </div>
      </div>
      <DataTable
        columns={[
          { key: 'model', label: 'Model' },
          { key: 't_review', label: 'Review ≥', num: true },
          { key: 't_block', label: 'Block ≥', num: true },
          { key: 'total_cost', label: 'Total cost', num: true, format: (v) => v.toLocaleString('en-US') },
          { key: 'fraud_missed', label: 'Fraud allowed', num: true },
          { key: 'legit_reviewed_pct', label: 'Legitimate sent to review', num: true, format: pct(2) },
          { key: 'legit_blocked', label: 'Legitimate blocked', num: true },
          { key: 'block_precision', label: 'Block precision', num: true, format: fixed(4) },
          { key: 'block_recall', label: 'Block recall', num: true, format: fixed(4) },
        ]}
        rows={block.operating_points}
      />
      <p className="note">
        MLP held-out PR-AUC across {sv.n_seeds} training seeds: mean {fixed(4)(sv.mean)}, standard deviation{' '}
        {fixed(4)(sv.std)}, range {fixed(4)(sv.min)} to {fixed(4)(sv.max)}. MLP training PR-AUC:{' '}
        {fixed(4)(block.train_pr_auc)}. Of the {bs.lgb_misses} held-out fraud cases LightGBM scored below 0.001, the
        MLP flags {bs.mlp_flags} at its review threshold and Random Forest flags {bs.rf_flags}.
      </p>
      <p className="source">Source: {block.source}</p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Cross-dataset validation on the ULB credit card data (notebook 06)
// ---------------------------------------------------------------------------
function CrossDataset({ block }) {
  if (!hasData(block)) return <Pending what="The cross-dataset validation" />
  return (
    <div className="grid cols-2">
      <ChartCard
        title="Same methodology on a second dataset: PR-AUC"
        subtitle={`${block.dataset} · ${block.split} · ${block.ci_method}`}
        source={block.source}
        note="A different dataset from the charts above. The models were trained on this dataset; the PaySim model was not reused."
        table={{ columns: CI_COLUMNS, rows: block.models }}
      >
        <CiBarChart models={block.models} labelWidth={170} />
      </ChartCard>
      <section className="card">
        <div className="card-head">
          <div className="titles">
            <h3>What the cross-dataset test showed</h3>
            <p>{block.dataset_detail}</p>
          </div>
        </div>
        <ul className="findings">
          {block.findings.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
        <p className="source">Source: {block.source}</p>
      </section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Other recorded results (figures only)
// ---------------------------------------------------------------------------
function Tile({ label, value, detail, source }) {
  return (
    <section className="card tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      <div className="detail">{detail}</div>
      <p className="source">Source: {source}</p>
    </section>
  )
}

function OtherResults({ metrics }) {
  const { period_pr_auc: periods, drift_psi: drift, blind_spot: blind, drift_stress_test: stress } = metrics
  const heldOut = hasData(periods) ? periods.periods.find((p) => p.period === 'held_out') : null
  const largest = hasData(stress) ? stress.additive.shifts[stress.additive.shifts.length - 1] : null
  return (
    <div className="grid cols-4">
      {heldOut && (
        <Tile
          label="Held-out PR-AUC, LightGBM"
          value={fixed(4)(heldOut.pr_auc)}
          detail={`95% CI [${fixed(4)(heldOut.ci_low)}, ${fixed(4)(heldOut.ci_high)}] · days ${heldOut.days}`}
          source={periods.source}
        />
      )}
      {hasData(drift) && (
        <Tile
          label="SHAP importance rank correlation"
          value={fixed(4)(drift.shap_rank_correlation.spearman_train_vs_held_out)}
          detail="Spearman, train vs held-out feature importance"
          source={drift.shap_rank_correlation.source}
        />
      )}
      {hasData(blind) && (
        <Tile
          label={`Blind spot, ${blind.model}`}
          value={`${blind.demo_sample.missed} / ${blind.demo_sample.fraud_cases}`}
          detail={`${blind.definition}, demo sample. Full held-out set: ${blind.held_out.missed} / ${blind.held_out.fraud_cases}.`}
          source={`${blind.demo_sample.source}; ${blind.held_out.source}`}
        />
      )}
      {largest && (
        <Tile
          label={`Drift stress test, ${stress.model}`}
          value={fixed(4)(largest.pr_auc)}
          detail={`PR-AUC with +${largest.shift.toLocaleString('en-US')} added to ${stress.features_shifted.join(' and ')}. Unshifted: ${fixed(4)(stress.baseline_pr_auc)}.`}
          source={stress.additive.source}
        />
      )}
    </div>
  )
}

export default function Validation({ modelInfo }) {
  const { data, error, loading } = useFetch(getValidationMetrics)
  if (loading) return <Loading />
  if (error) return <ApiError error={error} />
  const deployedModel = modelInfo ? MODEL_NAMES[modelInfo.model_class] || modelInfo.model_class : null

  return (
    <>
      <div className="page-head">
        <h2>Model Validation</h2>
        <p>
          Every figure on this page is transcribed from a saved notebook output and served from{' '}
          <code>api/validation_metrics.json</code>. Each chart names its source cell and has a table view of the same
          numbers.
          {deployedModel &&
            ` The API currently serves the ${deployedModel} model with thresholds ${modelInfo.thresholds.t_review} (review) and ${modelInfo.thresholds.t_block} (block).`}
        </p>
      </div>

      <div className="grid cols-2">
        <Progression block={data.pr_auc_progression} />
        <ModelComparison block={data.model_comparison} />
      </div>

      <h3 className="section-title">Model comparison at operating thresholds</h3>
      <OperatingPoints block={data.mlp_comparison} />

      <h3 className="section-title">Cross-dataset validation</h3>
      <CrossDataset block={data.cross_dataset} />

      <h3 className="section-title">Drift</h3>
      <Drift block={data.drift_psi} periodPrAuc={data.period_pr_auc} />

      <h3 className="section-title">Recall by transaction amount</h3>
      <Recall block={data.amount_stratified_recall} deployedModel={deployedModel} />

      <h3 className="section-title">Other recorded results</h3>
      <OtherResults metrics={data} />
    </>
  )
}
