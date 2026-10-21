import { useQueryClient } from '@tanstack/react-query'
import { Pause, Play, RotateCcw, Square } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { EmptyState, Kpi, Note, PageHeader, TierBadge } from '@/components/common'
import { Button, Card, Select, Table, Td, Th } from '@/components/ui'
import { api, streamUrl, type StreamEvent, type Tier } from '@/lib/api'
import { cn, fmt, sentence } from '@/lib/utils'

type Status = 'idle' | 'connecting' | 'live' | 'paused' | 'ended' | 'error'
const MAX_ROWS = 250
const SPEEDS = [1, 5, 20, 60]

export default function LiveFeed() {
  const [status, setStatus] = useState<Status>('idle')
  const [rows, setRows] = useState<StreamEvent[]>([])
  const [counts, setCounts] = useState<Record<Tier, number>>({ ALLOW: 0, REVIEW: 0, BLOCK: 0 })
  const [fraud, setFraud] = useState({ seen: 0, flagged: 0 })
  const [progress, setProgress] = useState({ position: 0, total: 0 })
  const [speed, setSpeed] = useState(5)
  const [error, setError] = useState<string | null>(null)
  const socket = useRef<WebSocket | null>(null)
  const paused = useRef(false)
  const buffer = useRef<StreamEvent[]>([])
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const take = useCallback((event: StreamEvent) => {
    setRows((current) => [event, ...current].slice(0, MAX_ROWS))
    setCounts((c) => ({ ...c, [event.tier]: c[event.tier] + 1 }))
    if (event.actual_fraud) setFraud((f) => ({ seen: f.seen + 1, flagged: f.flagged + (event.tier === 'ALLOW' ? 0 : 1) }))
    setProgress((p) => ({ ...p, position: event.position }))
  }, [])

  const stop = useCallback(() => {
    socket.current?.close()
    socket.current = null
  }, [])

  const start = useCallback((restart: boolean) => {
    stop()
    setError(null)
    setStatus('connecting')
    paused.current = false
    buffer.current = []
    if (restart) {
      setRows([])
      setCounts({ ALLOW: 0, REVIEW: 0, BLOCK: 0 })
      setFraud({ seen: 0, flagged: 0 })
    }
    const ws = new WebSocket(streamUrl(speed, restart))
    socket.current = ws
    ws.onmessage = (message) => {
      const event = JSON.parse(message.data)
      if (event.event === 'start') {
        setStatus('live')
        setProgress({ position: event.offset, total: event.total })
      } else if (event.event === 'transaction') {
        // While paused the server keeps scoring; rows wait here until resumed.
        if (paused.current) buffer.current.push(event)
        else take(event)
      } else if (event.event === 'end') {
        setStatus('ended')
      } else if (event.event === 'error') {
        setError(event.detail)
        setStatus('error')
      }
    }
    ws.onerror = () => { setError('The live feed connection failed. Is the API running?'); setStatus('error') }
    ws.onclose = (close) => {
      if (close.code === 4401) { setError('Your session has expired. Sign in again.'); setStatus('error') }
      setStatus((s) => (s === 'live' || s === 'paused' || s === 'connecting' ? 'idle' : s))
      queryClient.invalidateQueries({ queryKey: ['overview'] })
      queryClient.invalidateQueries({ queryKey: ['cases'] })
    }
  }, [speed, stop, take, queryClient])

  useEffect(() => stop, [stop])

  const togglePause = () => {
    if (status === 'live') { paused.current = true; setStatus('paused') }
    else if (status === 'paused') {
      paused.current = false
      buffer.current.forEach(take)
      buffer.current = []
      setStatus('live')
    }
  }

  // Clicking a REVIEW row opens its case.
  async function open(event: StreamEvent) {
    if (event.tier !== 'REVIEW') return
    const list = await api.cases({ status: 'all', q: event.assessment_id, limit: 1 })
    if (list.cases[0]) navigate(`/cases/${list.cases[0].id}`)
  }

  const running = status === 'live' || status === 'paused' || status === 'connecting'
  const total = counts.ALLOW + counts.REVIEW + counts.BLOCK

  return (
    <>
      <PageHeader title="Live feed"
        description="Held-out transactions (days 14 to 16) replayed in time order. Each one goes through the real scoring path: feature store lookup, model, calibration, tier."
        actions={<>
          <Select aria-label="Replay speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} disabled={running} className="w-36">
            {SPEEDS.map((s) => <option key={s} value={s}>{s} per second</option>)}
          </Select>
          {!running && <Button variant="primary" onClick={() => start(false)}><Play className="h-4 w-4" />{progress.position ? 'Continue' : 'Start'}</Button>}
          {running && <Button onClick={togglePause} disabled={status === 'connecting'}>
            {status === 'paused' ? <><Play className="h-4 w-4" />Resume</> : <><Pause className="h-4 w-4" />Pause</>}</Button>}
          {running && <Button onClick={stop}><Square className="h-4 w-4" />Stop</Button>}
          <Button onClick={() => start(true)} disabled={status === 'connecting'}><RotateCcw className="h-4 w-4" />Restart</Button>
        </>} />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Kpi label="Scored in this session" value={fmt.int(total)}
          hint={progress.total ? `position ${fmt.int(progress.position)} of ${fmt.int(progress.total)}` : 'not started'} />
        <Kpi label="Allowed" value={fmt.int(counts.ALLOW)} hint={total ? fmt.pct(counts.ALLOW / total, 1) : undefined} />
        <Kpi label="Sent to review" value={fmt.int(counts.REVIEW)} hint={total ? fmt.pct(counts.REVIEW / total, 1) : undefined} />
        <Kpi label="Blocked" value={fmt.int(counts.BLOCK)} hint={total ? fmt.pct(counts.BLOCK / total, 1) : undefined} />
        <Kpi label="Real fraud caught" value={`${fraud.flagged} of ${fraud.seen}`} hint="known only because this is a replay" />
      </div>

      <div className="mb-3">
        <Note>The replay oversamples fraud (about 1.2% of rows against a true rate of 0.07%) so that a short demo contains fraud. Rates on this page are not the model's real-world rates.</Note>
      </div>
      {error && <div className="mb-3"><Note tone="warning">{error}</Note></div>}

      <Card>
        <div className="flex items-center justify-between border-b px-4 py-2 text-xs text-secondary">
          <span className="inline-flex items-center gap-2">
            <span className={cn('h-2 w-2 rounded-full', status === 'live' ? 'animate-pulse bg-allow' : status === 'paused' ? 'bg-review' : 'bg-muted')} />
            {{ idle: 'Stopped', connecting: 'Connecting…', live: 'Live', paused: `Paused, ${buffer.current.length} waiting`, ended: 'Replay finished', error: 'Error' }[status]}
          </span>
          <span>Newest first · showing the last {MAX_ROWS} · click a review row to open its case</span>
        </div>
        {rows.length === 0 ? (
          <EmptyState title="No transactions yet" description="Press Start to begin the replay. Transactions sent to review appear in the review queue as they arrive." />
        ) : (
          <Table>
            <thead>
              <tr><Th>#</Th><Th>Decision</Th><Th className="text-right">Probability</Th><Th className="text-right">Anomaly</Th><Th>Type</Th>
                <Th className="text-right">Amount</Th><Th>Recipient</Th><Th>When (data time)</Th><Th>Truth</Th><Th>Main reason</Th></tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.assessment_id} onClick={() => open(row)}
                  className={cn(row.tier === 'REVIEW' && 'cursor-pointer hover:bg-surface-2')}>
                  <Td className="tabular text-muted">{row.position}</Td>
                  <Td><TierBadge tier={row.tier} /></Td>
                  <Td className="tabular text-right">{fmt.probability(row.fraud_probability)}</Td>
                  <Td className="tabular text-right text-secondary">{row.anomaly_score.toFixed(3)}</Td>
                  <Td className="text-secondary">{row.type}</Td>
                  <Td className="tabular text-right">{fmt.money(row.amount)}</Td>
                  <Td className="font-mono text-xs text-secondary">{row.name_dest}</Td>
                  <Td className="tabular whitespace-nowrap text-secondary">{fmt.step(row.step)}</Td>
                  <Td>{row.actual_fraud ? <span className="font-medium text-block-ink">Fraud</span> : <span className="text-muted">Legitimate</span>}</Td>
                  <Td className="max-w-xs truncate text-secondary">{row.reasons[0] ? sentence(row.reasons[0]) : ''}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </>
  )
}
