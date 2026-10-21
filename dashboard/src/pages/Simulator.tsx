import { useMutation, useQuery } from '@tanstack/react-query'
import { RotateCcw, Send } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Async, Legend, Note, PageHeader } from '@/components/common'
import { Button, Card, CardBody, CardHeader, Dialog, Slider, Textarea } from '@/components/ui'
import { api, type SimulatorData } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { CostModel, LOG_MIN, toPosition, toThreshold, type Outcome } from '@/lib/simulator'
import { cn, fmt } from '@/lib/utils'

type Split = 'eval_2' | 'held_out'
const SPLIT_LABEL: Record<Split, string> = { eval_2: 'eval_2 (days 11–13)', held_out: 'Held-out (days 14–16)' }
const logTick = (position: number) => (position >= 0 ? '1' : `1e${Math.round(position)}`)

function Control({ label, value, hint, children }: { label: string; value: string; hint: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium">{label}</span>
        <span className="tabular font-mono text-[13px]">{value}</span>
      </div>
      {children}
      <div className="mt-1 text-xs text-muted">{hint}</div>
    </div>
  )
}

// A figure with its change against the deployed setting. For every figure
// here, lower is better except precision and recall.
function Figure({ label, value, delta, higherIsBetter = false, sub }: { label: string; value: string; delta?: number; higherIsBetter?: boolean; sub?: string }) {
  const changed = delta !== undefined && Math.abs(delta) > 1e-9
  const better = changed && (delta! > 0) === higherIsBetter
  return (
    <Card className="p-4">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div className="tabular mt-1 text-2xl font-semibold tracking-tight">{value}</div>
      <div className={cn('tabular mt-1 text-xs', !changed ? 'text-muted' : better ? 'text-allow-ink' : 'text-block-ink')}>
        {sub ?? (changed ? `${delta! > 0 ? '▲' : '▼'} ${fmt.compact(Math.abs(delta!))} vs deployed (${better ? 'better' : 'worse'})` : 'same as deployed')}
      </div>
    </Card>
  )
}

function Workbench({ data, split }: { data: SimulatorData; split: Split }) {
  const { user } = useAuth()
  const deployed = data.deployed
  const model = useMemo(() => new CostModel(data), [data])
  const [review, setReview] = useState(toPosition(deployed.t_review))
  const [block, setBlock] = useState(deployed.t_block)
  const [costBlock, setCostBlock] = useState(deployed.cost_false_block)
  const [costReview, setCostReview] = useState(deployed.cost_review)
  const [proposing, setProposing] = useState(false)
  const [rationale, setRationale] = useState('')

  const reset = () => { setReview(toPosition(deployed.t_review)); setBlock(deployed.t_block); setCostBlock(deployed.cost_false_block); setCostReview(deployed.cost_review) }
  useEffect(reset, [data]) // eslint-disable-line react-hooks/exhaustive-deps

  const tReview = toThreshold(review)
  const tBlock = Math.max(block, tReview)
  const now: Outcome = useMemo(() => model.outcome(tReview, tBlock, costBlock, costReview), [model, tReview, tBlock, costBlock, costReview])
  const base: Outcome = useMemo(() => model.outcome(deployed.t_review, deployed.t_block, costBlock, costReview), [model, deployed, costBlock, costReview])

  // Cost as the review threshold sweeps its whole range, block threshold held.
  const curve = useMemo(() => {
    const top = toPosition(tBlock)
    return Array.from({ length: 221 }, (_, i) => {
      const position = LOG_MIN + ((top - LOG_MIN) * i) / 220
      const o = model.outcome(toThreshold(position), tBlock, costBlock, costReview)
      return { position, cost: Math.max(o.totalCost, 1), fraudAllowed: o.fraudAllowed, legitReviewedPct: o.legitReviewedPct }
    })
  }, [model, tBlock, costBlock, costReview])
  const cheapest = curve.reduce((best, p) => (p.cost < best.cost ? p : best), curve[0])
  const deployedPosition = toPosition(deployed.t_review)
  const changed = Math.abs(review - deployedPosition) > 1e-9 || Math.abs(block - deployed.t_block) > 1e-9

  const propose = useMutation({
    mutationFn: () => api.propose({
      t_review: tReview, t_block: tBlock, cost_false_block: costBlock, cost_review: costReview, split, rationale: rationale.trim(),
      projected: { total_cost: now.totalCost, fraud_allowed: now.fraudAllowed, legit_reviewed_pct: now.legitReviewedPct, legit_blocked: now.legitBlocked,
        block_precision: now.blockPrecision, block_recall: now.blockRecall },
    }),
    onSuccess: () => setRationale(''),
  })

  return (
    <div className="grid gap-4 xl:grid-cols-[20rem_1fr]">
      <Card className="self-start">
        <CardHeader title="Settings" action={<Button size="sm" variant="ghost" onClick={reset}><RotateCcw className="h-3.5 w-3.5" />Reset</Button>} />
        <CardBody className="space-y-5">
          <Control label="Review threshold" value={fmt.threshold(tReview)} hint="Below this a transaction is allowed. Logarithmic scale.">
            <Slider aria-label="Review threshold" min={LOG_MIN} max={0} step={0.01} value={[review]} onValueChange={([v]) => setReview(v)} />
          </Control>
          <Control label="Block threshold" value={tBlock.toFixed(3)} hint="At or above this a transaction is blocked automatically.">
            <Slider aria-label="Block threshold" min={0.05} max={1} step={0.005} value={[block]} onValueChange={([v]) => setBlock(v)} />
          </Control>
          <div className="border-t pt-4">
            <div className="mb-3 text-xs font-medium tracking-wide text-muted uppercase">Cost assumptions</div>
            <div className="space-y-5">
              <Control label="Wrongly blocking a legitimate payment" value={costBlock.toFixed(0)} hint="Cost units per wrongful block.">
                <Slider aria-label="Cost of a wrongful block" min={1} max={100} step={1} value={[costBlock]} onValueChange={([v]) => setCostBlock(v)} />
              </Control>
              <Control label="Sending a transaction to review" value={costReview.toFixed(1)} hint="Cost units per review. A missed fraud always costs its amount.">
                <Slider aria-label="Cost of a review" min={0.1} max={20} step={0.1} value={[costReview]} onValueChange={([v]) => setCostReview(v)} />
              </Control>
            </div>
          </div>
          {user?.role === 'admin' ? (
            <Button variant="primary" className="w-full" disabled={!changed} onClick={() => { propose.reset(); setProposing(true) }}>
              <Send className="h-4 w-4" />Propose these thresholds
            </Button>
          ) : <Note>Only an admin can propose a threshold change.</Note>}
        </CardBody>
      </Card>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Figure label="Total cost" value={fmt.int(now.totalCost)} delta={now.totalCost - base.totalCost} />
          <Figure label="Frauds let through" value={`${now.fraudAllowed} of ${model.nFraud}`} delta={now.fraudAllowed - base.fraudAllowed}
            sub={now.fraudAllowed ? `worth ${fmt.int(now.fraudAllowedAmount)}` : undefined} />
          <Figure label="Review workload" value={fmt.int(now.reviewWorkload)} delta={now.reviewWorkload - base.reviewWorkload}
            sub={`${fmt.pct(now.legitReviewedPct)} of legitimate traffic`} />
          <Figure label="Wrongful blocks" value={fmt.int(now.legitBlocked)} delta={now.legitBlocked - base.legitBlocked} />
          <Figure label="Block precision" value={fmt.pct(now.blockPrecision, 1)} delta={now.blockPrecision - base.blockPrecision} higherIsBetter
            sub={`${now.fraudBlocked} frauds among ${now.fraudBlocked + now.legitBlocked} blocked`} />
          <Figure label="Block recall" value={fmt.pct(now.blockRecall, 1)} delta={now.blockRecall - base.blockRecall} higherIsBetter
            sub={`${now.fraudBlocked} of ${model.nFraud} frauds blocked outright`} />
        </div>

        <Card>
          <CardHeader title="Total cost against the review threshold"
            description="Both axes are logarithmic. To the left the cost is review volume; to the right it is missed fraud, which is why the curve turns into a cliff."
            action={<Legend items={[{ label: 'Total cost', color: 'var(--series-1)' }]} />} />
          <CardBody>
            <div className="h-80">
              <ResponsiveContainer>
                <LineChart data={curve} margin={{ top: 24, right: 24, bottom: 4, left: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="position" type="number" domain={[LOG_MIN, 0]} ticks={[-7, -6, -5, -4, -3, -2, -1, 0]} tickFormatter={logTick}
                    tickLine={false} axisLine={false} label={{ value: 'Review threshold (calibrated probability)', position: 'insideBottom', offset: -2, fontSize: 11 }} height={36} />
                  <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={(v) => fmt.compact(v)} tickLine={false} axisLine={false} width={48} allowDataOverflow />
                  <Tooltip cursor={{ stroke: 'var(--text-muted)', strokeDasharray: '3 3' }} content={({ active, payload }) => {
                    if (!active || !payload?.length) return null
                    const p = payload[0].payload as (typeof curve)[number]
                    return (
                      <div className="rounded-md border bg-surface px-2.5 py-2 text-xs shadow-md">
                        <div className="mb-1 font-medium">Review at {fmt.threshold(toThreshold(p.position))}</div>
                        <div className="tabular">Total cost {fmt.int(p.cost)}</div>
                        <div className="tabular text-secondary">{p.fraudAllowed} frauds let through</div>
                        <div className="tabular text-secondary">{fmt.pct(p.legitReviewedPct)} of legitimate traffic reviewed</div>
                      </div>
                    )
                  }} />
                  <ReferenceLine x={deployedPosition} stroke="var(--text-muted)" strokeDasharray="4 3"
                    label={{ value: 'Deployed', position: 'top', fontSize: 11 }} />
                  <Line dataKey="cost" name="Total cost" type="stepAfter" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <ReferenceDot x={cheapest.position} y={cheapest.cost} r={4} fill="var(--surface)" stroke="var(--series-1)" strokeWidth={2}
                    label={{ value: 'Cheapest', position: 'bottom', fontSize: 11 }} />
                  <ReferenceDot x={review} y={Math.max(now.totalCost, 1)} r={6} fill="var(--text)" stroke="var(--surface)" strokeWidth={2}
                    label={{ value: 'Your setting', position: 'top', fontSize: 11 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-2 text-[13px] text-secondary">
              The cheapest point on this data is a review threshold of <b className="tabular">{fmt.threshold(toThreshold(cheapest.position))}</b> at a cost of{' '}
              <b className="tabular">{fmt.int(cheapest.cost)}</b>. The deployed threshold was deliberately placed to its left: it was chosen to stay safe if the
              calibrated odds are off by a factor of two, because the right-hand side of the minimum is a cliff.
            </p>
          </CardBody>
        </Card>

        <Note>
          {split === 'eval_2'
            ? 'This is the period the deployed thresholds were chosen on. A setting that looks best here is in-sample; switch to held-out to see how it would have done on unseen days.'
            : 'Held-out data. Use it to check a setting, not to choose one: a threshold tuned here is tuned on the test set, which is the mistake this project corrected.'}
          {' '}Frauds are exact; legitimate counts come from a 4,000-bin histogram of scores, so they can differ from the exact figures by a few transactions.
        </Note>
      </div>

      <Dialog open={proposing} onOpenChange={setProposing} title="Propose a threshold change"
        description="This writes a proposal to the audit log. It does not change the thresholds in production.">
        {propose.isSuccess ? (
          <div className="space-y-3">
            <Note>Proposal recorded in the audit log. Production thresholds are unchanged.</Note>
            <div className="flex justify-end"><Button onClick={() => setProposing(false)}>Close</Button></div>
          </div>
        ) : (
          <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); propose.mutate() }}>
            <dl className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
              <dt className="text-muted">Review threshold</dt><dd>{fmt.threshold(deployed.t_review)} → <b>{fmt.threshold(tReview)}</b></dd>
              <dt className="text-muted">Block threshold</dt><dd>{deployed.t_block.toFixed(3)} → <b>{tBlock.toFixed(3)}</b></dd>
              <dt className="text-muted">Projected cost ({SPLIT_LABEL[split]})</dt><dd>{fmt.int(base.totalCost)} → <b>{fmt.int(now.totalCost)}</b></dd>
              <dt className="text-muted">Frauds let through</dt><dd>{base.fraudAllowed} → <b>{now.fraudAllowed}</b></dd>
            </dl>
            <Textarea autoFocus required minLength={3} placeholder="Why should these thresholds replace the current ones?" value={rationale} onChange={(e) => setRationale(e.target.value)} />
            {propose.error && <div role="alert" className="text-[13px] text-block-ink">{propose.error.message}</div>}
            <div className="flex justify-end gap-2">
              <Button onClick={() => setProposing(false)}>Cancel</Button>
              <Button type="submit" variant="primary" disabled={propose.isPending || rationale.trim().length < 3}>{propose.isPending ? 'Recording…' : 'Record proposal'}</Button>
            </div>
          </form>
        )}
      </Dialog>
    </div>
  )
}

export default function Simulator() {
  const [split, setSplit] = useState<Split>('eval_2')
  const query = useQuery({ queryKey: ['simulator', split], queryFn: () => api.simulator(split), staleTime: Infinity })

  return (
    <>
      <PageHeader title="Threshold simulator"
        description="Move the two thresholds and the cost assumptions and see, instantly, what the deployed model would have cost. Everything is recomputed in the browser from precomputed scores."
        actions={
          <div className="flex rounded-md border bg-surface p-0.5" role="group" aria-label="Data period">
            {(Object.keys(SPLIT_LABEL) as Split[]).map((s) => (
              <button key={s} type="button" onClick={() => setSplit(s)} aria-pressed={split === s}
                className={cn('rounded px-2.5 py-1 text-[13px]', split === s ? 'bg-surface-2 font-medium' : 'text-secondary hover:text-text')}>{SPLIT_LABEL[s]}</button>
            ))}
          </div>} />
      <Async query={query} rows={8}>
        {(data) => (
          <>
            <p className="mb-4 text-[13px] text-secondary">
              Model <b>{data.model.name} {data.model.version}</b> on {fmt.int(data.n_rows)} transactions ({data.n_fraud} fraud). Deployed: review from{' '}
              <span className="tabular font-mono">{fmt.threshold(data.deployed.t_review)}</span>, block from <span className="tabular font-mono">{data.deployed.t_block.toFixed(3)}</span>,
              costing <span className="tabular">{fmt.int(data.deployed.total_cost)}</span> on this period.
            </p>
            <Workbench data={data} split={split} />
          </>
        )}
      </Async>
    </>
  )
}
