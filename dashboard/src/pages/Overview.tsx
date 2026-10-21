import { useQuery } from '@tanstack/react-query'
import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Async, ChartTooltip, EmptyState, Kpi, Legend, LevelBadge, PageHeader } from '@/components/common'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api } from '@/lib/api'
import { fmt } from '@/lib/utils'

const TIER_SERIES = [
  { key: 'ALLOW', label: 'Allow', color: 'var(--allow)' },
  { key: 'REVIEW', label: 'Review', color: 'var(--review)' },
  { key: 'BLOCK', label: 'Block', color: 'var(--block)' },
] as const

const EVENT_LABEL: Record<string, string> = {
  'assessment.block': 'Transaction blocked',
  'case.escalate': 'Case escalated',
  'model.activate': 'Serving model changed',
  'thresholds.propose': 'Threshold change proposed',
}

export default function Overview() {
  const query = useQuery({ queryKey: ['overview'], queryFn: api.overview, refetchInterval: 15_000 })

  return (
    <>
      <PageHeader title="Overview" description="What the scoring service has done, what is waiting for a person, and whether the model's inputs still look like its training data." />
      <Async query={query} rows={6}>
        {({ kpis, tier_mix, drift_alerts, recent_events }) => (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
              <Kpi label="Scored, last 24 hours" value={fmt.int(kpis.scored_last_24h)} hint={`${fmt.int(kpis.scored_total)} in total`} />
              <Kpi label="Block rate" value={fmt.pct(kpis.block_rate)} hint="of transactions scored in the last 24 hours" />
              <Kpi label="Review queue" value={fmt.int(kpis.review_queue)}
                hint={kpis.review_queue ? `oldest waiting ${fmt.duration(kpis.oldest_open_case_seconds)}` : 'nothing waiting'}>
                {kpis.escalated > 0 && <div className="mt-1 text-xs text-review-ink">{kpis.escalated} escalated to an admin</div>}
              </Kpi>
              <Kpi label="Override rate" value={fmt.pct(kpis.override_rate, 1)} hint="customers who proceeded after a warning" />
              <Kpi label="Active model" value={kpis.active_model.version} hint={kpis.active_model.name} />
              <Kpi label="Drift status" value={<LevelBadge level={kpis.drift_status} />}
                hint={<Link to="/monitoring" className="text-accent hover:underline">{drift_alerts.length} flagged inputs</Link>} />
            </div>

            <div className="grid gap-4 xl:grid-cols-3">
              <Card className="xl:col-span-2">
                <CardHeader title="Decision mix over time"
                  description="Each bar is a consecutive batch of scored transactions, oldest on the left. Allowed transactions are left out so the two tiers that need attention are readable."
                  action={<Legend items={TIER_SERIES.slice(1).map((s) => ({ label: s.label, color: s.color }))} />} />
                <CardBody>
                  {tier_mix.length === 0 ? (
                    <EmptyState title="Nothing scored yet" description="Start the live feed to replay held-out transactions through the model."
                      action={<Link to="/live" className="text-[13px] text-accent hover:underline">Open the live feed</Link>} />
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer>
                        <BarChart data={tier_mix.map((b) => ({ ...b, label: `${b.from}–${b.to}` }))} barCategoryGap={2}>
                          <CartesianGrid vertical={false} />
                          <XAxis dataKey="label" tickLine={false} axisLine={false} interval="preserveStartEnd" minTickGap={40} />
                          <YAxis allowDecimals={false} tickLine={false} axisLine={false} width={32} />
                          <Tooltip cursor={{ fill: 'var(--surface-2)' }}
                            content={<ChartTooltip labelFormatter={(l) => `Transactions ${l}`} formatter={(v) => fmt.int(v)} />} />
                          <Bar dataKey="REVIEW" name="Review" stackId="t" fill="var(--review)" stroke="var(--surface)" strokeWidth={1} isAnimationActive={false} />
                          <Bar dataKey="BLOCK" name="Block" stackId="t" fill="var(--block)" stroke="var(--surface)" strokeWidth={1} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardBody>
              </Card>

              <Card>
                <CardHeader title="Drift alerts" description="Inputs or scores whose distribution has moved away from the training window."
                  action={<Link to="/monitoring" className="inline-flex items-center gap-1 text-xs text-accent hover:underline">Details<ArrowRight className="h-3 w-3" /></Link>} />
                <CardBody className="pt-2">
                  {drift_alerts.length === 0 ? <EmptyState title="No drift alerts" /> : (
                    <Table>
                      <thead><tr><Th>Input and period</Th><Th className="text-right">PSI</Th><Th /></tr></thead>
                      <tbody>
                        {drift_alerts.map((a) => (
                          <tr key={`${a.period}-${a.name}`}>
                            <Td>
                              <div className="font-mono text-xs break-all">{a.name}</div>
                              <div className="text-xs text-muted">{a.period.replace('_', ' ')}</div>
                            </Td>
                            <Td className="tabular text-right">{a.value.toFixed(3)}</Td>
                            <Td className="text-right"><LevelBadge level={a.level} /></Td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  )}
                </CardBody>
              </Card>
            </div>

            <Card>
              <CardHeader title="Recent events" description="Blocks, escalations, model changes and threshold proposals."
                action={<Link to="/audit" className="inline-flex items-center gap-1 text-xs text-accent hover:underline">Audit log<ArrowRight className="h-3 w-3" /></Link>} />
              <CardBody className="pt-2">
                {recent_events.length === 0 ? <EmptyState title="No events yet" /> : (
                  <Table>
                    <thead><tr><Th>When</Th><Th>Event</Th><Th>By</Th><Th>Detail</Th></tr></thead>
                    <tbody>
                      {recent_events.map((e) => (
                        <tr key={e.id}>
                          <Td className="tabular whitespace-nowrap text-secondary">{fmt.time(e.at)}</Td>
                          <Td>{EVENT_LABEL[e.action] ?? e.action}</Td>
                          <Td className="text-secondary">{e.actor}</Td>
                          <Td className="tabular text-secondary">
                            {typeof e.detail.amount === 'number' && <>amount {fmt.money(e.detail.amount)} </>}
                            {typeof e.detail.fraud_probability === 'number' && <>· probability {fmt.probability(e.detail.fraud_probability)} </>}
                            {typeof e.detail.reason === 'string' && e.detail.reason}
                            {typeof e.detail.rationale === 'string' && e.detail.rationale}
                          </Td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </CardBody>
            </Card>
          </div>
        )}
      </Async>
    </>
  )
}
