import { useQuery } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'
import { ErrorState, Kpi, LevelBadge, LoadingState, Note, PageHeader } from '@/components/common'
import { Button, Card, CardBody, CardHeader } from '@/components/ui'
import { api, type Level } from '@/lib/api'
import { fmt } from '@/lib/utils'

// One label / value line inside a card.
function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b py-2 text-[13px] last:border-b-0">
      <span className="text-muted">{label}</span>
      <span className="tabular text-right font-medium">{children}</span>
    </div>
  )
}

function asLevel(status: string): Level {
  return status === 'ok' || status === 'warning' || status === 'alert' ? status : 'unknown'
}

export default function Status() {
  const query = useQuery({ queryKey: ['status'], queryFn: api.status, refetchInterval: 15_000 })

  return (
    <>
      <PageHeader title="System status" description="What the API is running right now: model, database, monitoring and recent activity. Refreshes every 15 seconds."
        actions={<Button variant="ghost" onClick={() => query.refetch()} disabled={query.isFetching}>
          <RefreshCw className="h-4 w-4" />{query.isFetching ? 'Checking…' : 'Check now'}</Button>} />

      {query.isPending ? <LoadingState rows={6} />
        : query.error ? <ErrorState error={query.error} onRetry={query.refetch} />
        : (() => {
          const s = query.data
          return (
            <div className="space-y-4">
              {s.status !== 'ok' && <Note tone="warning">The API is up but something needs attention. Check the database line below.</Note>}

              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Kpi label="API" value={<LevelBadge level={s.status === 'ok' ? 'ok' : 'warning'} label={s.status === 'ok' ? 'OK' : 'Degraded'} />}
                  hint={`up for ${fmt.duration(s.uptime_seconds)}`} />
                <Kpi label="Active model" value={s.model.version} hint={s.model.name} />
                <Kpi label="Scored in the last 24 hours" value={s.activity ? fmt.int(s.activity.transactions_scored_last_24h) : '–'} />
                <Kpi label="Open cases" value={s.activity ? fmt.int(s.activity.open_cases) : '–'} />
              </div>

              <div className="grid gap-4 lg:grid-cols-3">
                <Card>
                  <CardHeader title="Model" />
                  <CardBody className="pt-2">
                    <Row label="Version">{s.model.version}</Row>
                    <Row label="Review threshold">{fmt.threshold(s.model.thresholds.t_review)}</Row>
                    <Row label="Block threshold">{fmt.threshold(s.model.thresholds.t_block)}</Row>
                    <Row label="Features">{s.model.feature_count}</Row>
                    <Row label="Trained on days">{s.model.train_days ? s.model.train_days.join(', ') : '–'}</Row>
                    <Row label="Registered">{s.model.created_at ? fmt.time(s.model.created_at) : '–'}</Row>
                  </CardBody>
                </Card>

                <Card>
                  <CardHeader title="Database" />
                  <CardBody className="pt-2">
                    <Row label="Reachable"><LevelBadge level={s.database.ok ? 'ok' : 'alert'} label={s.database.ok ? 'Yes' : 'No'} /></Row>
                    <Row label="Backend">{s.database.backend}</Row>
                    <Row label="Response time">{`${s.database.latency_ms} ms`}</Row>
                  </CardBody>
                </Card>

                <Card>
                  <CardHeader title="Monitoring" />
                  <CardBody className="pt-2">
                    {s.monitoring ? (
                      <>
                        <Row label="Latest report">{s.monitoring.period}</Row>
                        <Row label="Drift"><LevelBadge level={asLevel(s.monitoring.status)} /></Row>
                        <Row label="Checked">{fmt.time(s.monitoring.checked_at)}</Row>
                      </>
                    ) : <p className="py-2 text-[13px] text-muted">No monitoring report has been imported yet.</p>}
                  </CardBody>
                </Card>
              </div>

              <Card>
                <CardHeader title="Service" />
                <CardBody className="grid gap-x-8 pt-2 sm:grid-cols-2">
                  <div>
                    <Row label="Environment">{s.environment}</Row>
                    <Row label="Demo mode">{s.demo_mode ? 'On' : 'Off'}</Row>
                    <Row label="API version">{s.api_version}</Row>
                  </div>
                  <div>
                    <Row label="Worker process">{s.worker_pid}</Row>
                    <Row label="Last checked">{fmt.time(s.checked_at)}</Row>
                  </div>
                </CardBody>
              </Card>

              {s.demo_mode && <Note>Demo mode is on: model rollback is turned off so visitors can't change the model for everyone.</Note>}
            </div>
          )
        })()}
    </>
  )
}
