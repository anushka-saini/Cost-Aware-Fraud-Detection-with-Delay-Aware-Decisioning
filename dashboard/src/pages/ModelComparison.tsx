import { useQuery } from '@tanstack/react-query'
import { Async, Legend, Note, PageHeader } from '@/components/common'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api, type ModelSummary, type ResultsSummary } from '@/lib/api'
import { cn, fmt } from '@/lib/utils'

const NEW_COLOR = 'var(--series-1)'
const OLD_COLOR = 'var(--series-2)'

// PR-AUC with its 95% interval, one row per model, on a shared axis. The old
// reported figure sits on the same row so the change is visible at a glance.
function IntervalChart({ models }: { models: ModelSummary[] }) {
  const values = models.flatMap((m) => [m.ci_low, m.ci_high, ...(m.old ? [m.old.ci_low, m.old.ci_high] : [])])
  const low = Math.floor(Math.min(...values) * 20) / 20
  const high = Math.ceil(Math.max(...values) * 20) / 20
  const x = (v: number) => ((v - low) / (high - low)) * 100
  const ticks = Array.from({ length: Math.round((high - low) / 0.05) + 1 }, (_, i) => low + i * 0.05)
  const ordered = [...models].sort((a, b) => b.pr_auc - a.pr_auc)

  return (
    <div>
      <div className="grid grid-cols-[11rem_1fr_12rem] gap-3 text-xs text-muted">
        <span />
        <div className="relative h-4">
          {ticks.map((t) => <span key={t} className="tabular absolute -translate-x-1/2" style={{ left: `${x(t)}%` }}>{t.toFixed(2)}</span>)}
        </div>
        <span className="text-right">New [95% CI]</span>
      </div>
      {ordered.map((m) => (
        <div key={m.key} className="grid grid-cols-[11rem_1fr_12rem] items-center gap-3 border-t py-2 text-[13px]">
          <span className="flex items-center gap-2 font-medium">
            {m.label}
            {m.deployed && <span className="rounded border border-accent/40 px-1 text-[10px] font-semibold tracking-wide text-accent uppercase">Deployed</span>}
          </span>
          <div className="relative h-9">
            {ticks.map((t) => <span key={t} className="absolute inset-y-0 w-px bg-border" style={{ left: `${x(t)}%` }} />)}
            {/* new: interval + filled dot */}
            <span className="absolute top-2.5 h-0.5 rounded" style={{ left: `${x(m.ci_low)}%`, width: `${x(m.ci_high) - x(m.ci_low)}%`, background: NEW_COLOR }} />
            <span className="absolute top-1 h-3.5 w-3.5 -translate-x-1/2 rounded-full ring-2 ring-surface" style={{ left: `${x(m.pr_auc)}%`, background: NEW_COLOR }}
              title={`New: ${m.pr_auc.toFixed(4)} [${m.ci_low.toFixed(4)}, ${m.ci_high.toFixed(4)}]`} />
            {/* old: interval + hollow dot, on the line below */}
            {m.old && <>
              <span className="absolute top-[26px] h-0.5 rounded opacity-70" style={{ left: `${x(m.old.ci_low)}%`, width: `${x(m.old.ci_high) - x(m.old.ci_low)}%`, background: OLD_COLOR }} />
              <span className="absolute top-[21px] h-3 w-3 -translate-x-1/2 rounded-full border-2 bg-surface" style={{ left: `${x(m.old.pr_auc)}%`, borderColor: OLD_COLOR }}
                title={`Previously reported: ${m.old.pr_auc.toFixed(4)} [${m.old.ci_low.toFixed(4)}, ${m.old.ci_high.toFixed(4)}]`} />
            </>}
          </div>
          <span className="tabular text-right">
            {m.pr_auc.toFixed(4)} <span className="text-muted">[{m.ci_low.toFixed(3)}, {m.ci_high.toFixed(3)}]</span>
            {m.old && <div className="text-xs text-muted">was {m.old.pr_auc.toFixed(4)}</div>}
          </span>
        </div>
      ))}
    </div>
  )
}

// Each seed's PR-AUC as a tick on a common axis: how much one training run can move the number.
function SeedStrip({ model, low, high }: { model: ModelSummary; low: number; high: number }) {
  const x = (v: number) => ((v - low) / (high - low)) * 100
  return (
    <div className="relative h-5 min-w-40">
      <span className="absolute top-2.5 right-0 left-0 h-px bg-border" />
      {model.seed_values.map((v, i) => (
        <span key={i} className="absolute top-1 h-3 w-0.5 -translate-x-1/2 rounded" style={{ left: `${x(v)}%`, background: NEW_COLOR }} title={`Seed ${i + 1}: ${v.toFixed(4)}`} />
      ))}
    </div>
  )
}

const Change = ({ now, before, lowerIsBetter = true, format }: { now: number; before: number; lowerIsBetter?: boolean; format: (v: number) => string }) => {
  const better = now === before ? null : (now < before) === lowerIsBetter
  return (
    <span className="tabular">
      <span className="text-muted">{format(before)} → </span>
      <span className={cn('font-medium', better === true && 'text-allow-ink', better === false && 'text-block-ink')}>{format(now)}</span>
    </span>
  )
}

function Body({ data }: { data: ResultsSummary }) {
  const models = data.models
  const label = (key: string) => models.find((m) => m.key === key)?.label ?? key
  const seedLow = Math.min(...models.filter((m) => m.key !== 'logreg').flatMap((m) => m.seed_values)) - 0.005
  const seedHigh = Math.max(...models.flatMap((m) => m.seed_values)) + 0.005
  const dl = data.deep_learning
  const ablation = data.balance_ablation

  return (
    <div className="space-y-4">
      <Note>{data.protocol} {data.features} Held-out period: {fmt.int(data.n_rows)} transactions, {data.n_fraud} fraud.</Note>

      <Card>
        <CardHeader title="Held-out PR-AUC with 95% confidence intervals"
          description="Bootstrap intervals from 200 resamples. Overlapping intervals are not a ranking; see the paired tests below."
          action={<Legend items={[{ label: 'Honest protocol', color: NEW_COLOR }, { label: 'Previously reported', color: OLD_COLOR }]} />} />
        <CardBody><IntervalChart models={models} /></CardBody>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader title="Is one model really better?" description="Paired bootstrap on the same resamples. A difference counts only if its interval excludes zero." />
          <CardBody className="pt-2">
            <Table>
              <thead><tr><Th>Comparison</Th><Th className="text-right">PR-AUC difference</Th><Th className="text-right">95% CI</Th><Th>Verdict</Th></tr></thead>
              <tbody>
                {data.paired_tests.filter((t) => t.a !== 'logreg').map((t) => (
                  <tr key={`${t.a}-${t.b}`}>
                    <Td>{label(t.a)} − {label(t.b)}</Td>
                    <Td className="tabular text-right">{fmt.signed(t.difference)}</Td>
                    <Td className="tabular text-right text-secondary">{fmt.signed(t.ci_low)} to {fmt.signed(t.ci_high)}</Td>
                    <Td>{t.significant ? <span className="font-medium">{label(t.difference > 0 ? t.a : t.b)} is better</span> : <span className="text-muted">No clear difference</span>}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Spread over 5 training seeds" description="Same data, same settings, different random seed. PR-AUC is steady; cost is not." />
          <CardBody className="pt-2">
            <Table>
              <thead><tr><Th>Model</Th><Th>PR-AUC per seed</Th><Th className="text-right">Mean ± std</Th><Th className="text-right">Cost, min to max</Th></tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.key}>
                    <Td className="font-medium">{m.label}</Td>
                    <Td>{m.key === 'logreg' ? <span className="text-muted">deterministic</span> : <SeedStrip model={m} low={seedLow} high={seedHigh} />}</Td>
                    <Td className="tabular text-right">{m.seed_mean.toFixed(4)} ± {m.seed_std.toFixed(4)}</Td>
                    <Td className="tabular text-right text-secondary">{fmt.compact(Math.min(...m.seed_costs))} to {fmt.compact(Math.max(...m.seed_costs))}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="mt-2 text-xs text-muted">Axis for the seed marks: {seedLow.toFixed(3)} to {seedHigh.toFixed(3)}. The deployed model is the average of the five seeds, which is what makes its cost stable.</p>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Cost and operating point on held-out" description="Each model at thresholds fixed on days 11–13. Missed fraud costs its amount, a wrongful block 5, a review 1." />
        <CardBody className="pt-2">
          <Table>
            <thead><tr><Th>Model</Th><Th className="text-right">Total cost</Th><Th className="text-right">Frauds let through</Th><Th className="text-right">Legitimate reviewed</Th>
              <Th className="text-right">Wrongful blocks</Th><Th className="text-right">Block precision</Th><Th className="text-right">Block recall</Th><Th>Class weight</Th><Th>Calibration</Th></tr></thead>
            <tbody>
              {[...models].sort((a, b) => a.total_cost - b.total_cost).map((m) => (
                <tr key={m.key} className={cn(m.deployed && 'bg-surface-2')}>
                  <Td className="font-medium">{m.label}{m.deployed && <span className="ml-2 text-[11px] text-accent">deployed</span>}</Td>
                  <Td className="tabular text-right font-medium">{fmt.int(m.total_cost)}</Td>
                  <Td className="tabular text-right">{m.fraud_allowed}</Td>
                  <Td className="tabular text-right">{fmt.pct(m.legit_reviewed_pct)}</Td>
                  <Td className="tabular text-right">{m.legit_blocked}</Td>
                  <Td className="tabular text-right">{fmt.pct(m.block_precision, 1)}</Td>
                  <Td className="tabular text-right">{fmt.pct(m.block_recall, 1)}</Td>
                  <Td className="tabular text-secondary">{String(m.weight)}</Td>
                  <Td className="text-secondary capitalize">{m.calibration}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Old numbers against honest numbers"
          description="The earlier figures chose class weights and thresholds by looking at the held-out set. Green and red mark whether the honest figure is better or worse than the one first reported." />
        <CardBody className="pt-2">
          <Table>
            <thead><tr><Th>Model</Th><Th>What was wrong before</Th><Th className="text-right">PR-AUC</Th><Th className="text-right">Total cost</Th>
              <Th className="text-right">Frauds let through</Th><Th className="text-right">Wrongful blocks</Th><Th className="text-right">Class weight</Th></tr></thead>
            <tbody>
              {models.filter((m) => m.old).map((m) => (
                <tr key={m.key}>
                  <Td className="font-medium">{m.label}{m.old!.deployed && <div className="text-[11px] font-normal text-muted">deployed before</div>}</Td>
                  <Td className="max-w-56 text-secondary">{m.old!.flaw}</Td>
                  <Td className="text-right"><Change now={m.pr_auc} before={m.old!.pr_auc} lowerIsBetter={false} format={(v) => v.toFixed(4)} /></Td>
                  <Td className="text-right"><Change now={m.total_cost} before={m.old!.total_cost} format={fmt.int} /></Td>
                  <Td className="text-right"><Change now={m.fraud_allowed} before={m.old!.fraud_allowed} format={String} /></Td>
                  <Td className="text-right"><Change now={m.legit_blocked} before={m.old!.legit_blocked} format={String} /></Td>
                  <Td className="tabular text-right text-secondary">{m.old!.weight} → {String(m.weight)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          <p className="mt-2 text-xs text-muted">Full tables, including the same comparison on the original 17 features: results/old_vs_new.md.</p>
        </CardBody>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        {dl && (
          <Card>
            <CardHeader title="Deep learning components" description="What the label-free autoencoder and the hybrid added on held-out." />
            <CardBody className="space-y-3 pt-2">
              <Table>
                <thead><tr><Th>Model</Th><Th className="text-right">PR-AUC [95% CI]</Th><Th className="text-right">5 seeds</Th></tr></thead>
                <tbody>
                  <tr><Td>Autoencoder alone (no labels)</Td>
                    <Td className="tabular text-right">{dl.autoencoder.seed_42.pr_auc.toFixed(4)} <span className="text-muted">[{dl.autoencoder.seed_42.ci_low.toFixed(3)}, {dl.autoencoder.seed_42.ci_high.toFixed(3)}]</span></Td>
                    <Td className="tabular text-right">{dl.autoencoder.seeds.mean.toFixed(4)} ± {dl.autoencoder.seeds.std.toFixed(4)}</Td></tr>
                  <tr><Td>{dl.hybrid.label}</Td>
                    <Td className="tabular text-right">{dl.hybrid.pr_auc.toFixed(4)} <span className="text-muted">[{dl.hybrid.ci_low.toFixed(3)}, {dl.hybrid.ci_high.toFixed(3)}]</span></Td>
                    <Td className="tabular text-right">{dl.hybrid.seed_mean.toFixed(4)} ± {dl.hybrid.seed_std.toFixed(4)}</Td></tr>
                </tbody>
              </Table>
              <ul className="list-disc space-y-1 pl-5 text-[13px] text-secondary">
                <li>With no labels, the autoencoder puts {fmt.pct(dl.autoencoder.seed_42.share_of_fraud_in_top_1pct, 1)} of fraud in its top 1% most anomalous transactions
                  (base rate {fmt.pct(dl.autoencoder.base_rate, 3)}), but its ranking varies a lot from seed to seed.</li>
                <li>Hybrid minus supervised: {fmt.signed(dl.hybrid_minus_supervised.difference)} (95% CI {fmt.signed(dl.hybrid_minus_supervised.ci_low)} to {fmt.signed(dl.hybrid_minus_supervised.ci_high)}),
                  {dl.hybrid_minus_supervised.significant ? ' a significant difference.' : ' not significant. The hybrid did not help.'} It let {dl.hybrid.fraud_allowed} frauds through.</li>
                <li>Of the {dl.supervised_blind_spot.n} frauds the supervised model is least sure about, {dl.supervised_blind_spot.in_top_5pct_most_anomalous} are in the
                  autoencoder's top 5% and {dl.supervised_blind_spot.in_top_10pct_most_anomalous} in its top 10%.</li>
              </ul>
            </CardBody>
          </Card>
        )}
        {ablation && (
          <Card>
            <CardHeader title="Dependence on PaySim's balance bookkeeping" description={`The deployed model family retrained without ${ablation.removed_features.length} balance features.`} />
            <CardBody className="space-y-3 pt-2">
              <Table>
                <thead><tr><Th>Features</Th><Th className="text-right">PR-AUC [95% CI]</Th><Th className="text-right">Legitimate reviewed</Th><Th className="text-right">Block recall</Th></tr></thead>
                <tbody>
                  {([['With balance features', ablation.with_balance_features], ['Without', ablation.without_balance_features]] as const).map(([name, row]) => (
                    <tr key={name}><Td>{name}</Td>
                      <Td className="tabular text-right">{row.pr_auc.toFixed(4)} <span className="text-muted">[{row.ci_low.toFixed(3)}, {row.ci_high.toFixed(3)}]</span></Td>
                      <Td className="tabular text-right">{fmt.pct(row.legit_reviewed_pct)}</Td>
                      <Td className="tabular text-right">{fmt.pct(row.block_recall, 1)}</Td></tr>
                  ))}
                </tbody>
              </Table>
              <Note tone="warning">
                Much of what the model has learned is a regularity the simulator itself produces. These results are an upper bound for this dataset, not an estimate of
                performance on real payments.
              </Note>
            </CardBody>
          </Card>
        )}
      </div>

      {data.rolling && (
        <Card>
          <CardHeader title="Does label delay matter?" description="Daily retraining on days 11–16 when labels arrive 0 to 3 days late. Mean of daily PR-AUC with 95% intervals." />
          <CardBody className="pt-2">
            <Table>
              <thead><tr><Th>Label delay</Th><Th className="text-right">Supervised</Th><Th className="text-right">Hybrid (+ anomaly score)</Th><Th className="text-right">Hybrid − supervised</Th><Th>Verdict</Th></tr></thead>
              <tbody>
                {data.rolling.by_delay.map((row) => (
                  <tr key={row.delay_days}>
                    <Td>{row.delay_days} {row.delay_days === 1 ? 'day' : 'days'}</Td>
                    <Td className="tabular text-right">{row.supervised.mean_daily_pr_auc.toFixed(4)} <span className="text-muted">[{row.supervised.ci_low.toFixed(3)}, {row.supervised.ci_high.toFixed(3)}]</span></Td>
                    <Td className="tabular text-right">{row.hybrid.mean_daily_pr_auc.toFixed(4)} <span className="text-muted">[{row.hybrid.ci_low.toFixed(3)}, {row.hybrid.ci_high.toFixed(3)}]</span></Td>
                    <Td className="tabular text-right">{fmt.signed(row.hybrid_minus_supervised.difference)}</Td>
                    <Td className="text-secondary">{!row.hybrid_minus_supervised.significant ? 'No clear difference' : row.hybrid_minus_supervised.difference > 0 ? 'Hybrid better' : 'Hybrid slightly worse'}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="mt-2 text-[13px] text-secondary">Delays of up to three days barely change performance on this data, and the label-free signal does not compensate for what little is lost. The delay-aware mechanism exists and is tested; no benefit is claimed for it.</p>
          </CardBody>
        </Card>
      )}
    </div>
  )
}

export default function ModelComparison() {
  const query = useQuery({ queryKey: ['summary'], queryFn: api.summary, staleTime: Infinity })
  return (
    <>
      <PageHeader title="Model comparison" description="Every model under the same protocol, with intervals, seed spread and the figures the project reported before the protocol was corrected." />
      <Async query={query} rows={8}>
        {(data) => data.available ? <Body data={data} /> : <Note tone="warning">Result files are missing ({data.missing_sections?.join(', ')}). Run scripts/run_protocol.py.</Note>}
      </Async>
    </>
  )
}
