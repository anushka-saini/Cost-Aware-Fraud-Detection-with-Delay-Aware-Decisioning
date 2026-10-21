import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table'
import { ArrowDown, Check, CornerUpRight, Search, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { EmptyState, ErrorState, LoadingState, PageHeader, StatusBadge } from '@/components/common'
import { Button, Card, Dialog, Input, Kbd, Select, Table, Td, Textarea, Th } from '@/components/ui'
import { api, type CaseRow } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { cn, fmt, sentence } from '@/lib/utils'

type Action = 'approve' | 'decline' | 'escalate'
const ACTION_LABEL: Record<Action, string> = { approve: 'Approve', decline: 'Decline', escalate: 'Escalate' }
const ACTION_HELP: Record<Action, string> = {
  approve: 'The transaction is legitimate. It is recorded as a not-fraud label for retraining.',
  decline: 'The transaction is fraud. It is recorded as a fraud label for retraining.',
  escalate: 'Pass the case to an admin. No label is recorded.',
}
const SORTS = [['risk', 'Highest risk'], ['amount', 'Largest amount'], ['age', 'Longest waiting'], ['newest', 'Newest']] as const
const STATUSES = ['open', 'escalated', 'approved', 'declined', 'all']
const PAGE = 50
const column = createColumnHelper<CaseRow>()

// Time in queue: grey under 15 minutes, amber up to an hour, red beyond.
function Waiting({ seconds, closed }: { seconds: number; closed: boolean }) {
  const tone = closed ? 'text-muted' : seconds > 3600 ? 'text-block-ink font-medium' : seconds > 900 ? 'text-review-ink' : 'text-secondary'
  return <span className={cn('tabular', tone)}>{fmt.duration(seconds)}</span>
}

export default function ReviewQueue() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [status, setStatus] = useState('open')
  const [sort, setSort] = useState('risk')
  const [type, setType] = useState('')
  const [minAmount, setMinAmount] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [cursor, setCursor] = useState(0)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [pending, setPending] = useState<{ action: Action; ids: number[] } | null>(null)
  const [note, setNote] = useState('')

  const params = { status, sort, type: type || undefined, q: search || undefined,
    min_amount: minAmount ? Number(minAmount) : undefined, limit: PAGE, offset: page * PAGE }
  const query = useQuery({ queryKey: ['cases', params], queryFn: () => api.cases(params), placeholderData: keepPreviousData, refetchInterval: 10_000 })
  const cases = useMemo(() => query.data?.cases ?? [], [query.data])

  useEffect(() => { setPage(0); setCursor(0); setSelected(new Set()) }, [status, sort, type, minAmount, search])
  useEffect(() => { if (cursor >= cases.length) setCursor(Math.max(0, cases.length - 1)) }, [cases.length, cursor])

  const decide = useMutation({
    mutationFn: ({ action, ids, text }: { action: Action; ids: number[]; text: string }) =>
      ids.length === 1 ? api.decide(ids[0], action, text).then(() => ({ decided: ids, failed: [] })) : api.decideMany(ids, action, text),
    onSuccess: () => {
      setPending(null); setNote(''); setSelected(new Set())
      queryClient.invalidateQueries({ queryKey: ['cases'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
  })

  const canAct = (row: CaseRow) => row.status === 'open' || (row.status === 'escalated' && user?.role === 'admin')
  const ask = (action: Action, ids: number[]) => { if (ids.length) { decide.reset(); setNote(''); setPending({ action, ids }) } }

  // Keyboard: J / K move, Enter opens, X selects, A / D / E decide.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement
      if (pending || event.metaKey || event.ctrlKey || event.altKey || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      const row = cases[cursor]
      const key = event.key.toLowerCase()
      if (key === 'j') setCursor((c) => Math.min(c + 1, cases.length - 1))
      else if (key === 'k') setCursor((c) => Math.max(c - 1, 0))
      else if (key === 'enter' && row) navigate(`/cases/${row.id}`)
      else if (key === 'x' && row) setSelected((s) => { const next = new Set(s); next.has(row.id) ? next.delete(row.id) : next.add(row.id); return next })
      else if ((key === 'a' || key === 'd' || key === 'e') && row) {
        const ids = (selected.size ? cases.filter((c) => selected.has(c.id)) : [row]).filter(canAct).map((c) => c.id)
        ask(key === 'a' ? 'approve' : key === 'd' ? 'decline' : 'escalate', ids)
      } else return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, cursor, selected, pending, user])

  useEffect(() => { document.getElementById(`case-row-${cursor}`)?.scrollIntoView({ block: 'nearest' }) }, [cursor])

  const columns = useMemo(() => [
    column.display({ id: 'select', header: () => (
      <input type="checkbox" aria-label="Select all on this page" checked={cases.length > 0 && cases.every((c) => selected.has(c.id))}
        onChange={(e) => setSelected(e.target.checked ? new Set(cases.map((c) => c.id)) : new Set())} />
    ), cell: ({ row }) => (
      <input type="checkbox" aria-label={`Select case ${row.original.id}`} checked={selected.has(row.original.id)}
        onClick={(e) => e.stopPropagation()}
        onChange={() => setSelected((s) => { const next = new Set(s); next.has(row.original.id) ? next.delete(row.original.id) : next.add(row.original.id); return next })} />
    ) }),
    column.accessor('id', { header: 'Case', cell: (c) => <span className="tabular text-secondary">#{c.getValue()}</span> }),
    column.accessor('fraud_probability', { header: 'Probability', meta: 'num', cell: (c) => <span className="tabular font-medium">{fmt.probability(c.getValue())}</span> }),
    column.accessor('amount', { header: 'Amount', meta: 'num', cell: (c) => <span className="tabular">{fmt.money(c.getValue())}</span> }),
    column.accessor('type', { header: 'Type', cell: (c) => <span className="text-secondary">{c.getValue() ?? '–'}</span> }),
    column.accessor('name_dest', { header: 'Recipient', cell: (c) => <span className="font-mono text-xs text-secondary">{c.getValue() ?? '–'}</span> }),
    column.accessor('reasons', { header: 'Main reason', cell: (c) => <span className="block max-w-sm truncate text-secondary">{c.getValue()[0] ? sentence(c.getValue()[0]) : '–'}</span> }),
    column.accessor('age_seconds', { header: 'In queue', meta: 'num', cell: (c) => <Waiting seconds={c.getValue()} closed={Boolean(c.row.original.closed_at)} /> }),
    column.accessor('status', { header: 'Status', cell: (c) => <StatusBadge status={c.getValue()} /> }),
  ], [cases, selected])
  const table = useReactTable({ data: cases, columns, getCoreRowModel: getCoreRowModel(), getRowId: (row) => String(row.id) })

  const chosen = cases.filter((c) => selected.has(c.id) && canAct(c)).map((c) => c.id)
  const counts = query.data?.counts_by_status ?? {}
  const total = query.data?.total ?? 0

  return (
    <>
      <PageHeader title="Review queue"
        description="Transactions the model was not sure enough about to allow or block. Every decision needs a note and becomes a training label."
        actions={<div className="flex items-center gap-1.5 text-xs text-muted">
          <Kbd>J</Kbd><Kbd>K</Kbd> move <Kbd>Enter</Kbd> open <Kbd>X</Kbd> select <Kbd>A</Kbd> approve <Kbd>D</Kbd> decline <Kbd>E</Kbd> escalate
        </div>} />

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex rounded-md border bg-surface p-0.5">
          {STATUSES.map((s) => (
            <button key={s} type="button" onClick={() => setStatus(s)}
              className={cn('rounded px-2.5 py-1 text-[13px] capitalize', status === s ? 'bg-surface-2 font-medium' : 'text-secondary hover:text-text')}>
              {s}{s !== 'all' && counts[s] ? <span className="tabular ml-1.5 text-muted">{counts[s]}</span> : null}
            </button>
          ))}
        </div>
        <Select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value)} className="w-40">
          {SORTS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </Select>
        <Select aria-label="Transaction type" value={type} onChange={(e) => setType(e.target.value)} className="w-36">
          <option value="">All types</option>
          {['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT'].map((t) => <option key={t} value={t}>{t}</option>)}
        </Select>
        <Input aria-label="Minimum amount" type="number" min={0} placeholder="Min amount" value={minAmount} onChange={(e) => setMinAmount(e.target.value)} className="w-32" />
        <div className="relative">
          <Search className="pointer-events-none absolute top-2.5 left-2.5 h-4 w-4 text-muted" />
          <Input aria-label="Search account or assessment" placeholder="Account or assessment id" value={search} onChange={(e) => setSearch(e.target.value)} className="w-60 pl-8" />
        </div>
        {(type || minAmount || search) && <Button variant="ghost" size="sm" onClick={() => { setType(''); setMinAmount(''); setSearch('') }}><X className="h-3.5 w-3.5" />Clear filters</Button>}
      </div>

      {selected.size > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2 rounded-md border bg-surface-2 px-3 py-2 text-[13px]">
          <span className="font-medium">{selected.size} selected</span>
          {chosen.length < selected.size && <span className="text-muted">({chosen.length} can be decided by you)</span>}
          <Button size="sm" variant="allow" disabled={!chosen.length} onClick={() => ask('approve', chosen)}><Check className="h-3.5 w-3.5" />Approve</Button>
          <Button size="sm" variant="block" disabled={!chosen.length} onClick={() => ask('decline', chosen)}><X className="h-3.5 w-3.5" />Decline</Button>
          <Button size="sm" variant="review" disabled={!chosen.length} onClick={() => ask('escalate', chosen)}><CornerUpRight className="h-3.5 w-3.5" />Escalate</Button>
          <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>Clear selection</Button>
        </div>
      )}

      <Card>
        {query.isPending ? <LoadingState rows={8} className="p-4" />
          : query.error ? <ErrorState error={query.error} onRetry={query.refetch} />
          : cases.length === 0 ? (
            <EmptyState title={status === 'open' ? 'The queue is empty' : 'No cases match'}
              description={status === 'open' ? 'Nothing is waiting for review. Start the live feed to score more transactions.' : 'Try a different status or clear the filters.'} />
          ) : (
            <Table>
              <thead>
                {table.getHeaderGroups().map((group) => (
                  <tr key={group.id}>
                    {group.headers.map((header) => (
                      <Th key={header.id} className={cn(header.column.columnDef.meta === 'num' && 'text-right')}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {((sort === 'risk' && header.id === 'fraud_probability') || (sort === 'amount' && header.id === 'amount') || (sort === 'age' && header.id === 'age_seconds'))
                          && <ArrowDown className="ml-1 inline h-3 w-3" aria-label="sorted descending" />}
                      </Th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row, index) => (
                  <tr key={row.id} id={`case-row-${index}`} onClick={() => navigate(`/cases/${row.original.id}`)}
                    aria-selected={index === cursor}
                    className={cn('cursor-pointer hover:bg-surface-2', index === cursor && 'bg-surface-2 shadow-[inset_2px_0_0_var(--accent)]')}>
                    {row.getVisibleCells().map((cell) => (
                      <Td key={cell.id} className={cn(cell.column.columnDef.meta === 'num' && 'text-right')}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </Td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        {total > PAGE && (
          <div className="flex items-center justify-between px-4 py-2 text-xs text-secondary">
            <span className="tabular">{page * PAGE + 1}–{Math.min((page + 1) * PAGE, total)} of {fmt.int(total)}</span>
            <div className="flex gap-2">
              <Button size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
              <Button size="sm" disabled={(page + 1) * PAGE >= total} onClick={() => setPage(page + 1)}>Next</Button>
            </div>
          </div>
        )}
      </Card>

      <Dialog open={pending !== null} onOpenChange={(open) => !open && setPending(null)}
        title={pending ? `${ACTION_LABEL[pending.action]} ${pending.ids.length === 1 ? `case #${pending.ids[0]}` : `${pending.ids.length} cases`}` : ''}
        description={pending ? ACTION_HELP[pending.action] : undefined}>
        {pending && (
          <form onSubmit={(e) => { e.preventDefault(); decide.mutate({ ...pending, text: note.trim() }) }} className="space-y-3">
            <Textarea autoFocus required minLength={3} placeholder="Why? This note is stored with the decision." value={note} onChange={(e) => setNote(e.target.value)} />
            {decide.error && <div role="alert" className="text-[13px] text-block-ink">{decide.error.message}</div>}
            {decide.data && decide.data.failed.length > 0 && <div role="alert" className="text-[13px] text-block-ink">{decide.data.failed.length} could not be decided.</div>}
            <div className="flex justify-end gap-2">
              <Button onClick={() => setPending(null)}>Cancel</Button>
              <Button type="submit" variant={pending.action === 'approve' ? 'allow' : pending.action === 'decline' ? 'block' : 'review'}
                disabled={decide.isPending || note.trim().length < 3}>{decide.isPending ? 'Saving…' : ACTION_LABEL[pending.action]}</Button>
            </div>
          </form>
        )}
      </Dialog>
    </>
  )
}
