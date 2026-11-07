import { useQuery } from '@tanstack/react-query'
import { ArrowRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Note, PageHeader, TierBadge } from '@/components/common'
import { Card, CardBody, CardHeader, Table, Td, Th } from '@/components/ui'
import { api } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { fmt } from '@/lib/utils'

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <Card>
    <CardHeader title={title} />
    <CardBody className="space-y-2 pt-0 text-[13px] leading-relaxed text-secondary">{children}</CardBody>
  </Card>
)

const PageLink = ({ to, children }: { to: string; children: ReactNode }) => (
  <Link to={to} className="inline-flex items-center gap-0.5 font-medium text-accent hover:underline">{children}<ArrowRight className="h-3 w-3" /></Link>
)

function Versions() {
  const query = useQuery({ queryKey: ['results-v3'], queryFn: api.v3, staleTime: Infinity })
  const history = query.data?.available ? query.data.history : null
  if (!history) return <p className="text-muted">{query.isPending ? 'Loading results…' : 'Results not available.'}</p>
  return (
    <Table>
      <thead><tr><Th>Version</Th><Th>Model</Th><Th className="text-right">Held-out PR-AUC</Th><Th className="text-right">Held-out cost</Th><Th>Honest?</Th></tr></thead>
      <tbody>
        {history.map((h) => (
          <tr key={h.version}>
            <Td className="font-medium">{h.version}</Td>
            <Td>{h.model}</Td>
            <Td className="tabular text-right">{h.pr_auc.toFixed(4)}</Td>
            <Td className="tabular text-right">{fmt.int(h.total_cost)}</Td>
            <Td className={h.honest ? 'text-secondary' : 'text-review-ink'}>{h.honest ? 'Yes' : 'No: tuned on the held-out days'}</Td>
          </tr>
        ))}
      </tbody>
    </Table>
  )
}

export default function About() {
  const { user } = useAuth()
  const admin = user?.role === 'admin'
  return (
    <>
      <PageHeader title="About this project" description="What the system does, how it was tested, and what it cannot tell you." />
      <div className="grid gap-4 xl:grid-cols-2">
        <Section title="What it does">
          <p>
            A fraud-decision service for mobile-money transactions. Every transaction gets one of three decisions:
            {' '}<TierBadge tier="ALLOW" />, <TierBadge tier="REVIEW" /> (a person looks at it) or <TierBadge tier="BLOCK" />.
          </p>
          <p>
            The decisions are set to keep the total cost low, not to maximise accuracy. A fraud that gets through costs its full amount, blocking an honest
            customer costs 5, and a manual review costs 1. Because a missed fraud is so expensive, the system reviews a few percent of traffic to make sure
            fraud is almost never simply allowed.
          </p>
          <p>
            The data is PaySim, a public simulation of a mobile-money service (6.36 million transactions over about a month). It is simulated, not real
            customer data.
          </p>
        </Section>

        <Section title="How a decision is made">
          <ol className="list-decimal space-y-1 pl-5">
            <li>The service looks up the recipient's recent history and computes 16 inputs (amount, type, balance changes, how unusual the amount is for this recipient, time of day).</li>
            <li>The model (V3: LightGBM, the average of 5 training runs) turns them into a score.</li>
            <li>The score is calibrated into a fraud probability.</li>
            <li>Two thresholds turn the probability into a decision, and the top reasons are attached for the analyst.</li>
            <li>Every review opens a case in the queue; every decision and change is written to the audit log.</li>
          </ol>
          <p>Try it on real held-out transactions: <PageLink to="/explain">Explainability</PageLink></p>
        </Section>

        <Section title="How it was tested">
          <p>The transactions are split by day, and each block of days has one job:</p>
          <Table>
            <thead><tr><Th>Days</Th><Th>Used for</Th></tr></thead>
            <tbody>
              <tr><Td className="tabular whitespace-nowrap">5–7</Td><Td>training the models</Td></tr>
              <tr><Td className="tabular whitespace-nowrap">8–10</Td><Td>tuning the models' settings (together with 5–7)</Td></tr>
              <tr><Td className="tabular whitespace-nowrap">11–13</Td><Td>choosing the model, calibrating the probabilities and setting the thresholds</Td></tr>
              <tr><Td className="tabular whitespace-nowrap">14–16</Td><Td>the held-out test: scored once at the end, under a rule written down beforehand</Td></tr>
            </tbody>
          </Table>
          <p>
            A model counts as "better" only when a paired bootstrap test says so after correcting for the number of comparisons; otherwise the result is
            called "comparable". Every number on this dashboard is read from <code className="font-mono text-xs">results/metrics.json</code>, together with the script that produced it.
          </p>
        </Section>

        <Section title="Versions">
          <p>
            V1 reported numbers that were tuned on the held-out days, so they were too good. V2 redid everything with the split above. V3 compared linear models,
            tree ensembles, neural networks, an autoencoder and a stacked ensemble on days 11–13, kept a tuned LightGBM, and replaced V2 only after beating it
            on the one-time held-out test.
          </p>
          <Versions />
          {admin && <p>Full comparison, intervals and curves: <PageLink to="/models">Model lab</PageLink></p>}
        </Section>

        <Section title="Limitations">
          <ul className="list-disc space-y-1 pl-5">
            <li><strong className="font-medium text-text">Simulated data.</strong> The strongest inputs are balance mismatches that come from how PaySim does its bookkeeping. Results here are an upper bound for this dataset, not a prediction for real payments.</li>
            <li><strong className="font-medium text-text">The held-out days are not brand new.</strong> V1 was tuned on them and V2 was scored on them. V3 made every choice on earlier days and scored them once, but they had been seen before.</li>
            <li><strong className="font-medium text-text">Few frauds.</strong> The held-out days contain fewer than a thousand frauds, so the confidence intervals are wide and small differences mean little.</li>
            <li><strong className="font-medium text-text">The costs are assumptions.</strong> 5 for a wrongful block and 1 for a review are chosen values; the threshold simulator shows how the decisions change if they are different.</li>
            <li><strong className="font-medium text-text">A known weak spot.</strong> If the balance inputs shift by a constant amount, some fraud gets through. Monitoring raises an alert when that happens{admin ? <> (see <PageLink to="/monitoring">Monitoring</PageLink>)</> : null}.</li>
            <li><strong className="font-medium text-text">Label delay.</strong> The system can handle fraud labels that arrive days late, but on this data the delay made almost no difference, so no benefit is claimed.</li>
          </ul>
        </Section>

        <Section title="How this was built">
          <p>
            The project was built by Anushka. An AI coding assistant (Claude Code) was used throughout to help write and review code, run experiments
            and draft documentation.
          </p>
          <p>
            Stack: Python (scikit-learn, LightGBM, PyTorch for the autoencoder), FastAPI with PostgreSQL or SQLite, and this dashboard in React and TypeScript.
          </p>
          <Note>The source code, the full write-up and the instructions to run it all with one command are in the project's README.</Note>
        </Section>
      </div>
    </>
  )
}
