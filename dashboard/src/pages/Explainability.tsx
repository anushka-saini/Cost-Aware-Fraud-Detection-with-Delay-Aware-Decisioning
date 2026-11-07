import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { Async, EmptyState, Legend, Note, PageHeader, TierBadge } from '@/components/common'
import { ShapWaterfall } from '@/components/ShapWaterfall'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api, type BreakdownRow, type ExplainabilityData, type FeatureImportance, type Scenario, type Tier } from '@/lib/api'
import { cn, fmt } from '@/lib/utils'

// Short names for the model's inputs.
const NAME: Record<string, string> = {
  origin_balance_error: "Sender's balance mismatch",
  destination_balance_error: "Recipient's balance mismatch",
  destination_balance_is_zero: "Recipient's balance is zero",
  destination_amount_deviation: "Amount unusual for this recipient",
  destination_avg_previous_amount: "Recipient's usual amount",
  destination_transactions_last_24h: "Recipient's transfers, last 24 hours",
  destination_transactions_last_7d: "Recipient's transfers, last 7 days",
  destination_is_first_transaction: 'First payment to this recipient',
  time_of_day: 'Time of day',
  amount: 'Amount',
  type_PAYMENT: 'Is a payment',
  type_TRANSFER: 'Is a transfer',
  type_CASH_OUT: 'Is a cash-out',
  type_CASH_IN: 'Is a cash-in',
  type_DEBIT: 'Is a debit',
}
// Blue means "away from fraud" in the waterfall, so fraud rows are orange here.
const FRAUD_COLOR = 'var(--series-2)'
const LEGIT_COLOR = 'var(--series-1)'

// ---------------------------------------------------------------------------
// Overall importance
// ---------------------------------------------------------------------------

function ImportanceBars({ rows }: { rows: FeatureImportance[] }) {
  const max = Math.max(...rows.flatMap((r) => [r.fraud_mean_abs, r.legit_mean_abs]))
  const bar = (value: number, color: string, title: string) => (
    <div className="flex items-center gap-2" title={title}>
      <div className="h-2.5 flex-1">
        <div className="h-full rounded-r" style={{ width: `${Math.max((value / max) * 100, 0.5)}%`, background: color }} />
      </div>
      <span className="tabular w-12 text-right text-xs text-secondary">{value.toFixed(2)}</span>
    </div>
  )
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.feature} className="grid grid-cols-[14rem_1fr] items-center gap-3 text-[13px]">
          <span className="truncate" title={r.feature}>{NAME[r.feature] ?? r.feature}</span>
          <div className="space-y-0.5">
            {bar(r.fraud_mean_abs, FRAUD_COLOR, `${NAME[r.feature] ?? r.feature}, fraud: ${r.fraud_mean_abs.toFixed(3)} on average (signed mean ${fmt.signed(r.fraud_mean, 3)})`)}
            {bar(r.legit_mean_abs, LEGIT_COLOR, `${NAME[r.feature] ?? r.feature}, legitimate: ${r.legit_mean_abs.toFixed(3)} on average (signed mean ${fmt.signed(r.legit_mean, 3)})`)}
          </div>
        </div>
      ))}
    </div>
  )
}

function Overall({ data }: { data: NonNullable<ExplainabilityData['global']> }) {
  const top = data.features[0]
  return (
    <Card>
      <CardHeader title="What V3 pays attention to"
        description={`Average size of each input's push on the score (SHAP, in ${data.units}), on days ${data.days[0]}–${data.days[data.days.length - 1]}: all ${fmt.int(data.n_fraud)} frauds and a random ${fmt.int(data.n_legit_sample)} legitimate transactions. Longer bar, bigger influence.`}
        action={<Legend items={[{ label: 'Fraud', color: FRAUD_COLOR }, { label: 'Legitimate', color: LEGIT_COLOR }]} />} />
      <CardBody className="space-y-3 pt-2">
        <ImportanceBars rows={data.features} />
        <Note>
          For fraud, the strongest input is {(NAME[top.feature] ?? top.feature).toLowerCase()}. The balance mismatches are bookkeeping that PaySim's simulator produces,
          so much of what the model has learned is specific to this dataset. The time-of-day input combines the two hour features.
        </Note>
        <p className="text-xs text-muted">Source: results/metrics.json, section v3_explainability ({data._source}).</p>
      </CardBody>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Guided scenarios
// ---------------------------------------------------------------------------

function tierFor(probability: number, review: number, block: number): Tier {
  return probability >= block ? 'BLOCK' : probability >= review ? 'REVIEW' : 'ALLOW'
}

function ScenarioView({ scenario }: { scenario: Scenario }) {
  const status = useQuery({ queryKey: ['status'], queryFn: api.status, staleTime: 60_000 })
  const body = { ...scenario.features, step: scenario.transaction.step }
  delete (body as Record<string, unknown>).hour_sin
  delete (body as Record<string, unknown>).hour_cos
  const explain = useQuery({ queryKey: ['explain', scenario.id], queryFn: () => api.explain(body), staleTime: Infinity })
  const t = scenario.transaction
  const thresholds = status.data?.model.thresholds

  return (
    <div className="grid gap-4 lg:grid-cols-[23rem_1fr]">
      <div className="space-y-3">
        <Table>
          <tbody>
            <tr><Td className="text-secondary">Type</Td><Td className="text-right">{t.type}</Td></tr>
            <tr><Td className="text-secondary">Amount</Td><Td className="tabular text-right font-medium">{fmt.money(t.amount)}</Td></tr>
            <tr><Td className="text-secondary">Sender balance</Td><Td className="tabular text-right whitespace-nowrap">{fmt.money(t.oldbalanceOrg)} → {fmt.money(t.newbalanceOrig)}</Td></tr>
            <tr><Td className="text-secondary">Recipient balance</Td><Td className="tabular text-right whitespace-nowrap">{fmt.money(t.oldbalanceDest)} → {fmt.money(t.newbalanceDest)}</Td></tr>
            <tr><Td className="text-secondary whitespace-nowrap">When (data time)</Td><Td className="tabular text-right whitespace-nowrap">{fmt.step(t.step)}</Td></tr>
            <tr><Td className="text-secondary">Recipient</Td><Td className="text-right font-mono text-xs">{t.nameDest}</Td></tr>
            <tr><Td className="text-secondary whitespace-nowrap">What it really was</Td>
              <Td className="text-right">{scenario.actual_fraud ? <span className="font-medium text-block-ink">Fraud</span> : 'Legitimate'}</Td></tr>
          </tbody>
        </Table>
        <p className="text-xs text-muted">How it was picked: {scenario.rule}. {fmt.int(scenario.group_size)} held-out transactions matched.</p>
      </div>

      <div>
        {explain.isPending && <div className="text-[13px] text-muted">Asking the model…</div>}
        {explain.error && <Note tone="warning">Could not explain this transaction: {String(explain.error)}</Note>}
        {explain.data && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <div>
                <div className="text-xs text-muted">Fraud probability</div>
                <div className="tabular text-2xl font-semibold">{fmt.probability(explain.data.fraud_probability)}</div>
              </div>
              {thresholds && (
                <div>
                  <div className="text-xs text-muted">Decision</div>
                  <TierBadge tier={tierFor(explain.data.fraud_probability, thresholds.t_review, thresholds.t_block)} className="mt-1" />
                </div>
              )}
              {thresholds && (
                <div className="text-xs text-secondary">
                  review from {fmt.threshold(thresholds.t_review)}, block from {thresholds.t_block}
                </div>
              )}
            </div>
            <ShapWaterfall explanation={explain.data} />
          </div>
        )}
      </div>
    </div>
  )
}

function Scenarios({ data }: { data: ExplainabilityData }) {
  const [params, setParams] = useSearchParams()
  if (!data.scenarios.length) {
    return <EmptyState title="No demo scenarios" description="Run scripts/v3_explainability.py to pick them from the held-out days." />
  }
  const current = data.scenarios.find((s) => s.id === params.get('scenario')) ?? data.scenarios[0]
  return (
    <Card>
      <CardHeader title="Guided scenarios: real held-out transactions"
        description="Five transactions from days 14–16, each picked by a fixed rule, including two the model got wrong or nearly wrong. The explanation is computed live by the API." />
      <CardBody className="space-y-4 pt-2">
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Scenarios">
          {data.scenarios.map((s, i) => (
            <button key={s.id} role="tab" aria-selected={s.id === current.id} onClick={() => setParams({ scenario: s.id })}
              className={cn('rounded-md border px-3 py-1.5 text-left text-[13px]',
                s.id === current.id ? 'border-accent/50 bg-surface-2 font-medium' : 'text-secondary hover:bg-surface-2')}>
              <span className="tabular mr-1.5 text-muted">{i + 1}.</span>{s.title}
            </button>
          ))}
        </div>
        <ScenarioView key={current.id} scenario={current} />
      </CardBody>
    </Card>
  )
}

// Where V3 catches fraud on the held-out days (scripts/v3_breakdown.py).
const TYPE_NAME: Record<string, string> = {
  CASH_IN: 'Cash-in', CASH_OUT: 'Cash-out', DEBIT: 'Debit', PAYMENT: 'Payment', TRANSFER: 'Transfer',
}

function BreakdownTable({ title, rows, groupLabel }: { title: string; rows: BreakdownRow[]; groupLabel: (r: BreakdownRow) => React.ReactNode }) {
  return (
    <div className="min-w-0">
      <h3 className="mb-1 text-[13px] font-medium">{title}</h3>
      <div className="overflow-x-auto">
        <Table>
          <thead><tr>
            <Th>Group</Th><Th className="text-right">Transactions</Th><Th className="text-right">Fraud</Th>
            <Th className="text-right">Recall</Th><Th className="text-right">Blocked at once</Th>
            <Th className="text-right">Review rate</Th><Th className="text-right">Block rate</Th>
          </tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.group}>
                <Td>{groupLabel(r)}</Td>
                <Td className="tabular text-right text-secondary">{fmt.int(r.n_rows)}</Td>
                <Td className="tabular text-right">{fmt.int(r.n_fraud)}</Td>
                <Td className="tabular text-right">{r.recall == null ? <span className="text-muted">no fraud</span> : fmt.pct(r.recall, 1)}</Td>
                <Td className="tabular text-right">{r.block_recall == null ? '–' : fmt.pct(r.block_recall, 1)}</Td>
                <Td className="tabular text-right">{fmt.pct(r.review_rate)}</Td>
                <Td className="tabular text-right">{fmt.pct(r.block_rate, 3)}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    </div>
  )
}

function Breakdown({ data }: { data: NonNullable<ExplainabilityData['breakdown']> }) {
  return (
    <Card>
      <CardHeader title={`Where V3 catches fraud: days ${data.days[0]}–${data.days[data.days.length - 1]}`}
        description={`${fmt.int(data.n_rows)} held-out transactions, ${data.n_fraud} fraud, split by type and by amount. This describes the existing held-out result; nothing was tuned.`} />
      <CardBody className="space-y-5 pt-2">
        <BreakdownTable title="By transaction type" rows={data.by_type} groupLabel={(r) => TYPE_NAME[r.group] ?? r.group} />
        <BreakdownTable title="By amount quartile" rows={data.by_amount_quartile}
          groupLabel={(r) => <>{r.group} <span className="text-xs text-muted">{fmt.int(r.amount_from)} to {fmt.int(r.amount_to)}</span></>} />
        <p className="text-xs text-muted">
          Recall: share of fraud sent to review or blocked. Blocked at once: share of fraud blocked without waiting for a person.
          Review and block rates: share of all transactions in the group. Quartiles are of {data.quartiles_of}. Fraud only happens in cash-outs and transfers.
          Check: {data.check}. Source: {data._source}.
        </p>
      </CardBody>
    </Card>
  )
}

export default function Explainability() {
  const query = useQuery({ queryKey: ['explainability'], queryFn: api.explainability, staleTime: Infinity })
  return (
    <>
      <PageHeader title="Explainability"
        description="Why the model decides what it decides: which inputs matter overall, and a walk through real transactions, one input at a time." />
      <Async query={query} rows={8}>
        {(data) => (
          <div className="space-y-4">
            <Scenarios data={data} />
            {data.global ? <Overall data={data.global} />
              : <Note tone="warning">The overall importance has not been computed. Run scripts/v3_explainability.py.</Note>}
            {data.breakdown && <Breakdown data={data.breakdown} />}
          </div>
        )}
      </Async>
    </>
  )
}
