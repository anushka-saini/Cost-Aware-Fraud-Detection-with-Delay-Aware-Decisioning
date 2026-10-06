import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getDecisionLog } from '../api.js'
import { amount as formatAmount, probability } from '../format.js'
import useFetch from '../useFetch.js'
import ChartCard, { DataTable, Legend, Segmented } from '../components/ChartCard.jsx'
import ChartTooltip from '../components/ChartTooltip.jsx'
import { ApiError, Loading } from '../components/States.jsx'

const TICK = { fill: 'var(--muted)', fontSize: 12 }
const AXIS_LINE = { stroke: 'var(--axis)' }
const TIERS = ['ALLOW', 'REVIEW', 'BLOCK']
const OUTCOMES = [
  { key: 'cancelled', label: 'Cancelled', color: 'var(--series-1)' },
  { key: 'proceeded', label: 'Proceeded anyway', color: 'var(--series-2)' },
]
// Tier is a status, so it uses the same status colors as the Live Scoring banner.
const TIER_SERIES = [
  { key: 'ALLOW', label: 'ALLOW', color: 'var(--good)' },
  { key: 'REVIEW', label: 'REVIEW', color: 'var(--warning)' },
  { key: 'BLOCK', label: 'BLOCK', color: 'var(--critical)' },
]

const rate = (part, whole) => (whole ? `${((part / whole) * 100).toFixed(1)}%` : '–')

function summarize(rows) {
  const count = (list) => ({
    total: list.length,
    proceeded: list.filter((r) => r.user_proceeded).length,
    cancelled: list.filter((r) => !r.user_proceeded).length,
  })

  const flagged = count(rows.filter((r) => r.tier !== 'ALLOW'))
  const byTier = TIERS.map((tier) => ({ tier, ...count(rows.filter((r) => r.tier === tier)) })).filter(
    (t) => t.total > 0 || t.tier !== 'ALLOW',
  )

  // Timestamps are written in UTC. Bucket by day once the log spans more than one day, by hour before that.
  const days = new Set(rows.map((r) => r.timestamp.slice(0, 10)))
  const byDay = days.size > 1
  const buckets = new Map()
  for (const r of [...rows].sort((a, b) => a.timestamp.localeCompare(b.timestamp))) {
    const key = byDay ? r.timestamp.slice(0, 10) : `${r.timestamp.slice(0, 13).replace('T', ' ')}:00`
    const bucket = buckets.get(key) || { bucket: key, cancelled: 0, proceeded: 0, ALLOW: 0, REVIEW: 0, BLOCK: 0, flagged: 0, overridden: 0 }
    bucket[r.user_proceeded ? 'proceeded' : 'cancelled'] += 1
    bucket[r.tier] += 1
    if (r.tier !== 'ALLOW') {
      bucket.flagged += 1
      if (r.user_proceeded) bucket.overridden += 1
    }
    buckets.set(key, bucket)
  }

  return { flagged, byTier, volume: [...buckets.values()], unit: byDay ? 'day' : 'hour' }
}

function Tile({ label, value, detail }) {
  return (
    <section className="card tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      <div className="detail">{detail}</div>
    </section>
  )
}

function StackedBars({ series = OUTCOMES, ...props }) {
  return series.map((o, i) => (
    <Bar
      key={o.key}
      dataKey={o.key}
      name={o.label}
      stackId="stack"
      fill={o.color}
      stroke="var(--surface)"
      strokeWidth={2}
      isAnimationActive={false}
      radius={i === series.length - 1 ? props.endRadius : 0}
      barSize={24}
    />
  ))
}

export default function DecisionLog() {
  const { data, error, loading, reload } = useFetch(getDecisionLog)
  const [breakdown, setBreakdown] = useState('By outcome')
  const rows = data?.rows
  const summary = useMemo(() => (rows?.length ? summarize(rows) : null), [rows])

  if (loading && !data) return <Loading />
  if (error) return <ApiError error={error} />

  const head = (
    <div className="page-head" style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
      <div style={{ flex: 1 }}>
        <h2>Decision Log</h2>
        <p>
          Read from <code>api/decision_log.csv</code>, which <code>/confirm-decision</code> appends to when a user
          cancels or proceeds on a REVIEW or BLOCK result. ALLOW results need no decision and are not logged.
        </p>
      </div>
      <button type="button" className="btn" onClick={reload}>
        Refresh
      </button>
    </div>
  )

  if (!summary) {
    return (
      <>
        {head}
        <div className="state">
          <strong>No decisions logged yet</strong>
          {data.exists
            ? 'The log file exists but has no rows.'
            : 'The log file is created the first time a decision is confirmed.'}{' '}
          Score a transaction that is flagged REVIEW or BLOCK in Live Scoring, choose Cancel or Proceed anyway, then
          refresh.
        </div>
      </>
    )
  }

  const { flagged, byTier, volume, unit } = summary
  const tier = (name) => byTier.find((t) => t.tier === name) || { total: 0, proceeded: 0 }
  const tierSeries = TIER_SERIES.filter((t) => byTier.some((b) => b.tier === t.key))
  const volumeSeries = breakdown === 'By tier' ? tierSeries : OUTCOMES
  const recent = [...rows].sort((a, b) => b.timestamp.localeCompare(a.timestamp)).slice(0, 15)

  return (
    <>
      {head}

      <div className="grid cols-4">
        <Tile label="Decisions logged" value={rows.length} detail={`${flagged.total} on REVIEW or BLOCK results`} />
        <Tile
          label="Override rate"
          value={rate(flagged.proceeded, flagged.total)}
          detail={`${flagged.proceeded} of ${flagged.total} flagged results proceeded anyway`}
        />
        <Tile
          label="Override rate, REVIEW"
          value={rate(tier('REVIEW').proceeded, tier('REVIEW').total)}
          detail={`${tier('REVIEW').proceeded} of ${tier('REVIEW').total}`}
        />
        <Tile
          label="Override rate, BLOCK"
          value={rate(tier('BLOCK').proceeded, tier('BLOCK').total)}
          detail={`${tier('BLOCK').proceeded} of ${tier('BLOCK').total}`}
        />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <ChartCard
          title="Decision volume over time"
          subtitle={`Logged decisions per ${unit} (UTC), ${breakdown === 'By tier' ? 'by tier' : 'by outcome'}`}
          controls={<Segmented options={['By outcome', 'By tier']} value={breakdown} onChange={setBreakdown} label="Breakdown" />}
          table={{
            columns: [
              { key: 'bucket', label: unit === 'day' ? 'Day (UTC)' : 'Hour (UTC)' },
              ...tierSeries.map((t) => ({ key: t.key, label: t.label, num: true })),
              { key: 'cancelled', label: 'Cancelled', num: true },
              { key: 'proceeded', label: 'Proceeded anyway', num: true },
              { key: 'rate', label: 'Override rate', num: true, format: (_, r) => rate(r.overridden, r.flagged) },
            ],
            rows: volume,
          }}
        >
          <Legend items={volumeSeries} />
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={volume} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="bucket" tick={TICK} axisLine={AXIS_LINE} tickLine={false} minTickGap={16} />
              <YAxis allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={36} />
              <Tooltip
                cursor={{ fill: 'var(--surface-2)' }}
                content={<ChartTooltip note={(d) => `Override rate ${rate(d.overridden, d.flagged)} (${d.overridden} of ${d.flagged} flagged)`} />}
              />
              {StackedBars({ series: volumeSeries, endRadius: [4, 4, 0, 0] })}
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Tier distribution of logged decisions"
          subtitle="Number of logged decisions per tier, by outcome"
          table={{
            columns: [
              { key: 'tier', label: 'Tier' },
              { key: 'total', label: 'Logged', num: true },
              { key: 'cancelled', label: 'Cancelled', num: true },
              { key: 'proceeded', label: 'Proceeded anyway', num: true },
              { key: 'rate', label: 'Override rate', num: true, format: (_, r) => rate(r.proceeded, r.total) },
            ],
            rows: byTier,
          }}
        >
          <Legend items={OUTCOMES} />
          <ResponsiveContainer width="100%" height={byTier.length * 56 + 48}>
            <BarChart layout="vertical" data={byTier} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid horizontal={false} stroke="var(--grid)" />
              <XAxis type="number" allowDecimals={false} tick={TICK} axisLine={AXIS_LINE} tickLine={false} />
              <YAxis type="category" dataKey="tier" width={70} tick={{ ...TICK, fill: 'var(--text-2)' }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: 'var(--surface-2)' }}
                content={<ChartTooltip note={(d) => `${d.total} logged · override rate ${rate(d.proceeded, d.total)}`} />}
              />
              {StackedBars({ endRadius: [0, 4, 4, 0] })}
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <h3 className="section-title">Most recent decisions</h3>
      <section className="card">
        <DataTable
          columns={[
            { key: 'timestamp', label: 'Time (UTC)', format: (v) => v.slice(0, 19).replace('T', ' ') },
            { key: 'tier', label: 'Tier' },
            { key: 'fraud_probability', label: 'Fraud probability', num: true, format: probability },
            { key: 'amount', label: 'Amount', num: true, format: formatAmount },
            { key: 'user_proceeded', label: 'Decision', format: (v) => (v ? 'Proceeded anyway' : 'Cancelled') },
          ]}
          rows={recent}
        />
        {rows.length > recent.length && (
          <p className="note">
            Showing the {recent.length} most recent of {rows.length} logged decisions.
          </p>
        )}
      </section>
    </>
  )
}
