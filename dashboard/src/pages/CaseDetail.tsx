import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Check, CornerUpRight, X } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Async, EmptyState, Note, PageHeader, StatusBadge, TierBadge } from '@/components/common'
import { ShapWaterfall } from '@/components/ShapWaterfall'
import { Button, Card, CardBody, CardHeader, Table, Td, Textarea, Th } from '@/components/ui'
import { api, type CaseDetail as Case } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { cn, fmt, sentence } from '@/lib/utils'

type Action = 'approve' | 'decline' | 'escalate'

const Field = ({ label, children }: { label: string; children: ReactNode }) => (
  <div className="flex items-baseline justify-between gap-4 border-b py-1.5 last:border-0">
    <dt className="text-[13px] text-muted">{label}</dt>
    <dd className="tabular text-right text-[13px]">{children}</dd>
  </div>
)

// Where a score sits between 0 and its thresholds, on a log scale so that a
// probability of 0.0002 and one of 0.6 are both visible.
function ScoreScale({ value, tReview, tBlock }: { value: number; tReview: number; tBlock: number }) {
  const floor = Math.min(tReview / 100, 1e-6)
  const x = (v: number) => (Math.log10(Math.max(v, floor)) - Math.log10(floor)) / (0 - Math.log10(floor)) * 100
  return (
    <div className="mt-3">
      <div className="relative h-2 rounded-full bg-surface-2">
        <span className="absolute inset-y-0 rounded-l-full bg-allow/50" style={{ left: 0, width: `${x(tReview)}%` }} />
        <span className="absolute inset-y-0 bg-review/60" style={{ left: `${x(tReview)}%`, width: `${x(tBlock) - x(tReview)}%` }} />
        <span className="absolute inset-y-0 rounded-r-full bg-block/60" style={{ left: `${x(tBlock)}%`, right: 0 }} />
        <span className="absolute -top-1 h-4 w-1 -translate-x-1/2 rounded bg-text ring-2 ring-surface" style={{ left: `${x(value)}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-muted">
        <span>allow below {fmt.threshold(tReview)}</span><span>block from {fmt.threshold(tBlock)}</span>
      </div>
    </div>
  )
}

function ActionPanel({ data }: { data: Case }) {
  const { user } = useAuth()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [note, setNote] = useState('')
  const [action, setAction] = useState<Action | null>(null)
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['case', data.id] })
    queryClient.invalidateQueries({ queryKey: ['cases'] })
    queryClient.invalidateQueries({ queryKey: ['overview'] })
  }
  const decide = useMutation({ mutationFn: (a: Action) => api.decide(data.id, a, note.trim()), onSuccess: () => { setNote(''); refresh() } })
  const addNote = useMutation({ mutationFn: () => api.addNote(data.id, note.trim()), onSuccess: () => { setNote(''); refresh() } })

  const escalated = data.status === 'escalated'
  const actionable = data.status === 'open' || (escalated && user?.role === 'admin')
  const ready = note.trim().length >= 3
  const submit = (a: Action) => { setAction(a); decide.mutate(a) }

  return (
    <Card>
      <CardHeader title="Decision" description={actionable ? 'A note is required. Approve and decline become training labels.' : undefined} />
      <CardBody className="space-y-3">
        {!actionable && (
          <Note tone={escalated ? 'warning' : 'neutral'}>
            {escalated ? 'Escalated: waiting for an admin to resolve it.' : <>This case was <b>{data.status}</b> by {data.decided_by}.</>}
          </Note>
        )}
        <Textarea aria-label="Note" placeholder={actionable ? 'Why? Required for a decision.' : 'Add a note to the case'} value={note} onChange={(e) => setNote(e.target.value)} />
        {(decide.error || addNote.error) && <div role="alert" className="text-[13px] text-block-ink">{(decide.error ?? addNote.error)!.message}</div>}
        {actionable && (
          <div className="grid grid-cols-3 gap-2">
            <Button variant="allow" disabled={!ready || decide.isPending} onClick={() => submit('approve')}><Check className="h-4 w-4" />Approve</Button>
            <Button variant="block" disabled={!ready || decide.isPending} onClick={() => submit('decline')}><X className="h-4 w-4" />Decline</Button>
            <Button variant="review" disabled={!ready || decide.isPending || escalated} onClick={() => submit('escalate')}><CornerUpRight className="h-4 w-4" />Escalate</Button>
          </div>
        )}
        <div className="flex items-center justify-between">
          <Button size="sm" variant="ghost" disabled={!note.trim() || addNote.isPending} onClick={() => addNote.mutate()}>Save as note only</Button>
          {decide.isSuccess && action && <button type="button" className="text-xs text-accent hover:underline" onClick={() => navigate('/queue')}>Saved. Back to the queue</button>}
        </div>
      </CardBody>
    </Card>
  )
}

export default function CaseDetail() {
  const id = Number(useParams().id)
  const query = useQuery({ queryKey: ['case', id], queryFn: () => api.caseDetail(id), enabled: Number.isFinite(id) })

  return (
    <>
      <Link to="/queue" className="mb-3 inline-flex items-center gap-1 text-[13px] text-secondary hover:text-text"><ArrowLeft className="h-3.5 w-3.5" />Review queue</Link>
      <Async query={query} rows={8}>
        {(data) => {
          const t = data.transaction
          const s = data.scores
          // Seeded history only has the amount and time; show the other columns when any row has them.
          const detailed = data.recipient_history.some((h) => h.type || h.name_orig)
          return (
            <>
              <PageHeader title={`Case #${data.id}`}
                description={<span className="inline-flex flex-wrap items-center gap-2"><StatusBadge status={data.status} /><TierBadge tier={data.tier} />
                  <span>opened {fmt.time(data.opened_at)} · {data.closed_at ? `closed after ${fmt.duration(data.age_seconds)}` : `waiting ${fmt.duration(data.age_seconds)}`}</span></span>} />
              <div className="grid gap-4 xl:grid-cols-3">
                <div className="space-y-4 xl:col-span-2">
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardHeader title="Fraud probability" description={`Supervised model ${s.model_version}, calibrated`} />
                      <CardBody>
                        <div className="tabular text-3xl font-semibold tracking-tight">{fmt.probability(s.fraud_probability)}</div>
                        <ScoreScale value={s.fraud_probability} tReview={s.t_review} tBlock={s.t_block} />
                        <div className="mt-2 text-xs text-muted">Raw ensemble score {s.raw_score.toFixed(4)}</div>
                      </CardBody>
                    </Card>
                    <Card>
                      <CardHeader title="Anomaly score" description="Label-free autoencoder: how unlike ordinary traffic this looks" />
                      <CardBody>
                        <div className="tabular text-3xl font-semibold tracking-tight">{s.anomaly_score == null ? '–' : s.anomaly_score.toFixed(3)}</div>
                        {s.anomaly_score != null && s.anomaly_review_threshold != null && (
                          <div className="mt-3 text-[13px] text-secondary">
                            {s.anomaly_score >= s.anomaly_review_threshold ? 'Above' : 'Below'} the top-1% line ({s.anomaly_review_threshold.toFixed(3)}).
                            {' '}This score only changes a decision when the supervised model's labels are out of date.
                          </div>
                        )}
                        {s.fallback_applied && <div className="mt-2"><Note tone="warning">Sent to review by the anomaly fallback; the supervised model would have allowed it.</Note></div>}
                      </CardBody>
                    </Card>
                  </div>

                  <Card>
                    <CardHeader title="Why it was flagged" />
                    <CardBody className="space-y-4">
                      {data.reasons.length > 0 && (
                        <ol className="space-y-1 text-sm">
                          {data.reasons.map((reason, i) => (
                            <li key={reason} className="flex gap-2"><span className="tabular text-muted">{i + 1}.</span>{sentence(reason)}</li>
                          ))}
                        </ol>
                      )}
                      {data.explanation ? <ShapWaterfall explanation={data.explanation} />
                        : <Note>Feature contributions are not available: this case was scored by a model version that is no longer active.</Note>}
                    </CardBody>
                  </Card>

                  <Card>
                    <CardHeader title="Recipient's earlier transactions"
                      description={t ? `Most recent first, from the feature store, for ${t.name_dest}` : undefined} />
                    <CardBody className="pt-2">
                      {data.recipient_history.length === 0 ? <EmptyState title="No earlier transactions" description="This is the first transaction the service has seen to this recipient." /> : (
                        <Table>
                          <thead><tr><Th>When (data time)</Th>{detailed && <><Th>Type</Th><Th>From</Th></>}<Th className="text-right">Amount</Th></tr></thead>
                          <tbody>
                            {data.recipient_history.map((h) => (
                              <tr key={h.transaction_id}>
                                <Td className="tabular text-secondary">{fmt.step(h.step)}</Td>
                                {detailed && <><Td className="text-secondary">{h.type ?? '–'}</Td>
                                  <Td className="font-mono text-xs text-secondary">{h.name_orig ?? '–'}</Td></>}
                                <Td className="tabular text-right">{fmt.money(h.amount)}</Td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      )}
                    </CardBody>
                  </Card>

                  <Card>
                    <CardHeader title="Audit trail" description="Everything recorded about this case and its assessment" />
                    <CardBody className="pt-2">
                      <Table>
                        <thead><tr><Th>When</Th><Th>Action</Th><Th>By</Th><Th>Note</Th></tr></thead>
                        <tbody>
                          {data.audit_trail.map((e) => (
                            <tr key={e.id}>
                              <Td className="tabular whitespace-nowrap text-secondary">{fmt.time(e.at)}</Td>
                              <Td className="font-mono text-xs">{e.action}</Td>
                              <Td className="text-secondary">{e.actor} <span className="text-muted">({e.role})</span></Td>
                              <Td className="text-secondary">{typeof e.detail.note === 'string' ? e.detail.note : ''}</Td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </CardBody>
                  </Card>
                </div>

                <div className="space-y-4">
                  <ActionPanel data={data} />
                  <Card>
                    <CardHeader title="Transaction" />
                    <CardBody className="pt-2">
                      {t ? (
                        <dl>
                          <Field label="Amount"><span className="font-semibold">{fmt.money(t.amount)}</span></Field>
                          <Field label="Type">{t.type}</Field>
                          <Field label="When (data time)">{fmt.step(t.step)}</Field>
                          <Field label="From"><span className="font-mono text-xs">{t.name_orig ?? '–'}</span></Field>
                          <Field label="To"><span className="font-mono text-xs">{t.name_dest}</span></Field>
                          <Field label="Sender balance">{fmt.money(t.old_balance_orig)} → {fmt.money(t.new_balance_orig)}</Field>
                          <Field label="Recipient balance">{fmt.money(t.old_balance_dest)} → {fmt.money(t.new_balance_dest)}</Field>
                          <Field label="Assessment"><span className="font-mono text-xs">{data.assessment_id}</span></Field>
                        </dl>
                      ) : <Note>Scored from pre-computed features, so the raw transaction was not stored.</Note>}
                    </CardBody>
                  </Card>
                  <Card>
                    <CardHeader title="Similar past cases" description="Same type, similar amount, closest score" />
                    <CardBody className="pt-2">
                      {data.similar_cases.length === 0 ? <div className="py-3 text-[13px] text-muted">No closed cases like this one yet.</div> : (
                        <ul className="divide-y">
                          {data.similar_cases.map((c) => (
                            <li key={c.id}>
                              <Link to={`/cases/${c.id}`} className="flex items-center justify-between gap-2 py-2 text-[13px] hover:text-accent">
                                <span className="tabular">#{c.id} · {fmt.money(c.amount)}</span>
                                <span className="flex items-center gap-2"><span className="tabular text-muted">{fmt.probability(c.fraud_probability)}</span><StatusBadge status={c.status} /></span>
                              </Link>
                            </li>
                          ))}
                        </ul>
                      )}
                    </CardBody>
                  </Card>
                  <Card>
                    <CardHeader title="Notes" />
                    <CardBody className="pt-2">
                      {data.notes.length === 0 ? <div className="py-3 text-[13px] text-muted">No notes yet.</div> : (
                        <ul className="space-y-3">
                          {data.notes.map((n, i) => (
                            <li key={i} className={cn('text-[13px]')}>
                              <div>{n.note}</div>
                              <div className="text-xs text-muted">{n.author} · {fmt.time(n.created_at)}</div>
                            </li>
                          ))}
                        </ul>
                      )}
                    </CardBody>
                  </Card>
                </div>
              </div>
            </>
          )
        }}
      </Async>
    </>
  )
}
