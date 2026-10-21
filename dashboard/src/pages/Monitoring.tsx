import { useQuery } from '@tanstack/react-query'
import { ArrowUpCircle, MinusCircle } from 'lucide-react'
import { Async, EmptyState, Kpi, LevelBadge, Note, PageHeader } from '@/components/common'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api, type DriftReport, type Level, type ResultsSummary } from '@/lib/api'
import { cn, fmt } from '@/lib/utils'

const CELL: Record<Level, string> = {
  ok: 'text-secondary',
  warning: 'bg-review-soft text-review-ink font-medium',
  alert: 'bg-block-soft text-block-ink font-semibold',
  unknown: 'text-muted',
}
const PERIOD_LABEL: Record<string, string> = { eval_1: 'Days 8–10', eval_2: 'Days 11–13', held_out: 'Days 14–16' }

// Features (rows) by period (columns). The number is always shown; the tint
// only repeats what the number and the threshold already say.
function PsiHeatmap({ reports }: { reports: DriftReport[] }) {
  const names = reports[0].psi.map((p) => p.name)
  const worst = (name: string) => Math.max(...reports.map((r) => r.psi.find((p) => p.name === name)?.psi ?? 0))
  const ordered = [...names].sort((a, b) => worst(b) - worst(a))
  const kind = (name: string) => reports[0].psi.find((p) => p.name === name)!.kind
  return (
    <Table>
      <thead>
        <tr><Th>Input</Th><Th>Kind</Th>{reports.map((r) => <Th key={r.period} className="text-right">{PERIOD_LABEL[r.period] ?? r.period}</Th>)}</tr>
      </thead>
      <tbody>
        {ordered.map((name) => (
          <tr key={name}>
            <Td className="font-mono text-xs">{name}</Td>
            <Td className="text-muted">{kind(name) === 'score' ? 'model output' : 'feature'}</Td>
            {reports.map((r) => {
              const cell = r.psi.find((p) => p.name === name)
              return (
                <Td key={r.period} className="p-1 text-right">
                  {cell ? <span className={cn('tabular inline-block min-w-16 rounded px-2 py-1', CELL[cell.level])}
                    title={`${name}, ${r.period}: PSI ${cell.psi.toFixed(4)} (${cell.level})`}>{cell.psi.toFixed(3)}</span> : '–'}
                </Td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </Table>
  )
}

function ChampionChallenger({ rolling }: { rolling: NonNullable<ResultsSummary['rolling']> }) {
  const delays = Object.keys(rolling.champion_challenger)
  return (
    <div className="space-y-4">
      {delays.map((delay) => {
        const policy = rolling.champion_challenger[delay]
        return (
          <div key={delay}>
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
              <span className="font-medium">Label delay: {delay} {delay === '1' ? 'day' : 'days'}</span>
              <span className="tabular text-secondary">
                {policy.n_promotions} promotions · mean daily PR-AUC {policy.mean_daily_pr_auc.toFixed(4)} (retraining every day: {policy.always_retrain_mean_daily_pr_auc.toFixed(4)})
              </span>
            </div>
            <Table>
              <thead><tr><Th>Scoring day</Th><Th>Decision</Th><Th>Challenger trained through</Th><Th>Checked on</Th>
                <Th className="text-right">PR-AUC difference [95% CI]</Th><Th>Serving model</Th><Th className="text-right">PR-AUC that day</Th></tr></thead>
              <tbody>
                {policy.decisions.map((d) => (
                  <tr key={d.scoring_day}>
                    <Td className="tabular">Day {d.scoring_day}</Td>
                    <Td>
                      {d.decision === 'promote'
                        ? <span className="inline-flex items-center gap-1 font-medium text-accent"><ArrowUpCircle className="h-3.5 w-3.5" />Promote</span>
                        : <span className="inline-flex items-center gap-1 text-secondary"><MinusCircle className="h-3.5 w-3.5" />Keep</span>}
                    </Td>
                    <Td className="tabular text-secondary">day {d.challenger_trained_through_day}</Td>
                    <Td className="tabular text-secondary">day {d.validation_day}</Td>
                    <Td className="tabular text-right text-secondary">
                      {d.pr_auc_difference === undefined ? <span className="text-muted">{d.reason}</span>
                        : `${fmt.signed(d.pr_auc_difference)} [${fmt.signed(d.ci_low!)}, ${fmt.signed(d.ci_high!)}]`}
                    </Td>
                    <Td className="tabular text-secondary">trained through day {d.serving_model_trained_through_day}</Td>
                    <Td className="tabular text-right">{d.serving_pr_auc_on_scoring_day.toFixed(4)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        )
      })}
    </div>
  )
}

export default function Monitoring() {
  const drift = useQuery({ queryKey: ['drift'], queryFn: api.drift, refetchInterval: 30_000 })
  const summary = useQuery({ queryKey: ['summary'], queryFn: api.summary, staleTime: Infinity })

  return (
    <>
      <PageHeader title="Model monitoring"
        description="Is the data the model sees still like the data it was trained on, are its decisions arriving at the expected rates, and what happens when the inputs it depends on move." />
      <div className="space-y-4">
        <Async query={drift} rows={6}>
          {(data) => (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Kpi label="Drift status" value={<LevelBadge level={data.status} />} hint={`${data.reports.length} period reports`} />
                <Kpi label="Review rate, live" value={fmt.pct(data.live.review_rate)} hint={`${fmt.int(data.live.assessments)} transactions scored by this service`} />
                <Kpi label="Block rate, live" value={fmt.pct(data.live.block_rate, 3)} />
                <Kpi label="Override rate, live" value={fmt.pct(data.live.override_rate, 1)}
                  hint={`${data.live.customer_decisions} customer decisions after a warning · alert above 30%`} />
              </div>

              <Card>
                <CardHeader title="Population Stability Index by period"
                  description="Each period compared with the training window (days 5–7). 0.10 to 0.25 is a warning; above 0.25 is an alert." />
                <CardBody className="pt-2">
                  {data.reports.length === 0
                    ? <EmptyState title="No drift reports yet" description="Run scripts/run_monitoring.py, then restart the service to import the reports." />
                    : <>
                      <PsiHeatmap reports={data.reports} />
                      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-secondary">
                        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-5 rounded bg-review-soft" />warning, 0.10 to 0.25</span>
                        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-5 rounded bg-block-soft" />alert, above 0.25</span>
                      </div>
                      <div className="mt-3">
                        <Note>
                          Two recipient-activity counts drift steadily, while the model score and the anomaly score barely move. The model makes little use of
                          those two features, so these alerts are a prompt to look, not a sign the model is failing.
                        </Note>
                      </div>
                    </>}
                </CardBody>
              </Card>

              {data.reports.length > 0 && (
                <Card>
                  <CardHeader title="Operating rates by period" description="Alert when a rate leaves 0.5× to 2× the rate observed on days 11–13, where the thresholds were set." />
                  <CardBody className="pt-2">
                    <Table>
                      <thead><tr><Th>Period</Th><Th>Rate</Th><Th className="text-right">Observed</Th><Th className="text-right">Expected band</Th><Th /></tr></thead>
                      <tbody>
                        {data.reports.flatMap((r) => r.rates.map((rate) => (
                          <tr key={`${r.period}-${rate.name}`}>
                            <Td className="text-secondary">{PERIOD_LABEL[r.period] ?? r.period}</Td>
                            <Td>{rate.name === 'review_rate' ? 'Sent to review' : rate.name === 'block_rate' ? 'Blocked' : rate.name}</Td>
                            <Td className="tabular text-right">{fmt.pct(rate.observed, 3)}</Td>
                            <Td className="tabular text-right text-secondary">{rate.band_low === undefined ? '–' : `${fmt.pct(rate.band_low, 3)} to ${fmt.pct(rate.band_high, 3)}`}</Td>
                            <Td className="text-right"><LevelBadge level={rate.level} /></Td>
                          </tr>
                        )))}
                      </tbody>
                    </Table>
                  </CardBody>
                </Card>
              )}
            </>
          )}
        </Async>

        <Async query={summary} rows={5}>
          {(data) => !data.available ? <Note>Result files are missing: {data.missing_sections?.join(', ')}.</Note> : (
            <>
              {data.stress_test && (
                <Card>
                  <CardHeader title="Stress test: drift in the features the model relies on"
                    description="The two balance-error features are shifted on the held-out period and the deployed model re-scored with its thresholds unchanged." />
                  <CardBody className="pt-2">
                    <Table>
                      <thead><tr><Th>Scenario</Th><Th className="text-right">PR-AUC</Th><Th className="text-right">Total cost</Th><Th className="text-right">Frauds let through</Th>
                        <Th className="text-right">Legitimate reviewed</Th><Th>Would monitoring alert?</Th></tr></thead>
                      <tbody>
                        <tr>
                          <Td>No shift</Td><Td className="tabular text-right">{data.stress_test.baseline.pr_auc.toFixed(4)}</Td>
                          <Td className="tabular text-right">{fmt.int(data.stress_test.baseline.total_cost)}</Td>
                          <Td className="tabular text-right">{data.stress_test.baseline.fraud_allowed}</Td>
                          <Td className="tabular text-right">{fmt.pct(data.stress_test.baseline.legit_reviewed_pct)}</Td><Td className="text-muted">not applicable</Td>
                        </tr>
                        {data.stress_test.scenarios.map((s) => (
                          <tr key={`${s.kind}-${s.value}`}>
                            <Td>{s.kind === 'multiply' ? `Multiply both by ${s.value}` : `Add ${fmt.int(s.value)} to both`}</Td>
                            <Td className="tabular text-right">{s.pr_auc.toFixed(4)}</Td>
                            <Td className="tabular text-right">{fmt.int(s.total_cost)}</Td>
                            <Td className={cn('tabular text-right', s.fraud_allowed > 0 && 'font-semibold text-block-ink')}>{s.fraud_allowed}</Td>
                            <Td className="tabular text-right">{fmt.pct(s.legit_reviewed_pct)}</Td>
                            <Td>{s.would_alert ? <LevelBadge level="alert" label="Yes: PSI alert" /> : <span className="text-muted">No</span>}</Td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                    <div className="mt-3">
                      <Note tone="warning">
                        A constant offset in these two features lets about a fifth of fraud through. In every such scenario the PSI alert fires and the review rate
                        falls far below its band, so the failure would be visible here before it was visible in losses.
                      </Note>
                    </div>
                  </CardBody>
                </Card>
              )}
              {data.rolling && (
                <Card>
                  <CardHeader title="Champion / challenger history"
                    description="Replay of daily retraining on days 11–16. A retrained model replaces the serving one only when it is significantly better on the most recent fully labelled day." />
                  <CardBody><ChampionChallenger rolling={data.rolling} /></CardBody>
                </Card>
              )}
            </>
          )}
        </Async>
      </div>
    </>
  )
}
