import { keepPreviousData, useMutation, useQuery } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, Download, Search, X } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { EmptyState, ErrorState, LoadingState, PageHeader } from '@/components/common'
import { Button, Card, Input, Select, Table, Td, Th } from '@/components/ui'
import { api } from '@/lib/api'
import { fmt } from '@/lib/utils'

const PAGE = 100
const GROUPS = [['', 'All actions'], ['case.', 'Case decisions and notes'], ['assessment.', 'Flagged assessments'], ['customer.', 'Customer choices'],
  ['auth.', 'Sign-ins'], ['model.', 'Model changes'], ['thresholds.', 'Threshold proposals'], ['label.', 'Labels']] as const

function summary(detail: Record<string, unknown>) {
  const parts: string[] = []
  if (typeof detail.amount === 'number') parts.push(`amount ${fmt.money(detail.amount)}`)
  if (typeof detail.fraud_probability === 'number') parts.push(`probability ${fmt.probability(detail.fraud_probability)}`)
  if (typeof detail.from === 'string' && typeof detail.to === 'string') parts.push(`${detail.from} → ${detail.to}`)
  if (typeof detail.t_review === 'number' && typeof detail.t_block === 'number') parts.push(`review ${fmt.threshold(detail.t_review)}, block ${detail.t_block}`)
  for (const key of ['note', 'reason', 'rationale']) if (typeof detail[key] === 'string') parts.push(detail[key] as string)
  return parts.join(' · ')
}

export default function AuditLog() {
  const [action, setAction] = useState('')
  const [actor, setActor] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [open, setOpen] = useState<number | null>(null)
  const filters = { action: action || undefined, actor: actor || undefined, q: search || undefined }
  const query = useQuery({ queryKey: ['audit', filters, page], queryFn: () => api.audit({ ...filters, limit: PAGE, offset: page * PAGE }), placeholderData: keepPreviousData })
  useEffect(() => setPage(0), [action, actor, search])

  const download = useMutation({
    mutationFn: () => api.auditCsv(filters),
    onSuccess: (blob) => {
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'audit_log.csv'
      link.click()
      URL.revokeObjectURL(url)
    },
  })
  const total = query.data?.total ?? 0
  const filtered = Boolean(action || actor || search)

  return (
    <>
      <PageHeader title="Audit log" description="An append-only record of every decision, flag, sign-in, model change and threshold proposal."
        actions={<Button onClick={() => download.mutate()} disabled={download.isPending || total === 0}>
          <Download className="h-4 w-4" />{download.isPending ? 'Preparing…' : `Export ${filtered ? 'filtered ' : ''}CSV`}</Button>} />

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Select aria-label="Action" value={action} onChange={(e) => setAction(e.target.value)} className="w-56">
          {GROUPS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </Select>
        <Input aria-label="Actor" placeholder="Actor (exact username)" value={actor} onChange={(e) => setActor(e.target.value)} className="w-48" />
        <div className="relative">
          <Search className="pointer-events-none absolute top-2.5 left-2.5 h-4 w-4 text-muted" />
          <Input aria-label="Search" placeholder="Search id, actor or action" value={search} onChange={(e) => setSearch(e.target.value)} className="w-64 pl-8" />
        </div>
        {filtered && <Button variant="ghost" size="sm" onClick={() => { setAction(''); setActor(''); setSearch('') }}><X className="h-3.5 w-3.5" />Clear</Button>}
        <span className="tabular ml-auto text-xs text-muted">{fmt.int(total)} entries</span>
      </div>
      {download.error && <div role="alert" className="mb-3 text-[13px] text-block-ink">{download.error.message}</div>}

      <Card>
        {query.isPending ? <LoadingState rows={10} className="p-4" />
          : query.error ? <ErrorState error={query.error} onRetry={query.refetch} />
          : query.data.rows.length === 0 ? (
            <EmptyState title={filtered ? 'No entries match' : 'The audit log is empty'} description={filtered ? 'Try different filters.' : 'Entries appear as transactions are flagged and decisions are made.'} />
          ) : (
            <Table>
              <thead><tr><Th className="w-8" /><Th>When</Th><Th>Actor</Th><Th>Role</Th><Th>Action</Th><Th>Entity</Th><Th>Summary</Th></tr></thead>
              <tbody>
                {query.data.rows.map((row) => (
                  <Fragment key={row.id}>
                    <tr onClick={() => setOpen(open === row.id ? null : row.id)} className="cursor-pointer hover:bg-surface-2" aria-expanded={open === row.id}>
                      <Td>{open === row.id ? <ChevronDown className="h-3.5 w-3.5 text-muted" /> : <ChevronRight className="h-3.5 w-3.5 text-muted" />}</Td>
                      <Td className="tabular whitespace-nowrap text-secondary">{fmt.time(row.at)}</Td>
                      <Td className="font-medium">{row.actor}</Td>
                      <Td className="text-muted">{row.role}</Td>
                      <Td className="font-mono text-xs">{row.action}</Td>
                      <Td className="font-mono text-xs text-secondary">{row.entity_type} {row.entity_id}</Td>
                      <Td className="max-w-md truncate text-secondary">{summary(row.detail)}</Td>
                    </tr>
                    {open === row.id && (
                      <tr><Td /><Td colSpan={6} className="bg-surface-2">
                        <pre className="overflow-x-auto font-mono text-xs whitespace-pre-wrap text-secondary">{JSON.stringify(row.detail, null, 2)}</pre>
                        <div className="mt-1 text-xs text-muted">Entry #{row.id}{row.request_id && <> · request {row.request_id}</>}</div>
                      </Td></tr>
                    )}
                  </Fragment>
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
    </>
  )
}
