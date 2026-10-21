import { AlertTriangle, Ban, CheckCircle2, CircleAlert, CircleCheck, Eye, Inbox, RefreshCw, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import type { Level, Tier } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button, Card, Skeleton } from './ui'

// Tier colours are reserved for these three decisions and never appear
// without the icon and the word.
const TIER = {
  ALLOW: { icon: CheckCircle2, className: 'bg-allow-soft text-allow-ink border-allow/30' },
  REVIEW: { icon: Eye, className: 'bg-review-soft text-review-ink border-review/40' },
  BLOCK: { icon: Ban, className: 'bg-block-soft text-block-ink border-block/30' },
} as const

export function TierBadge({ tier, className }: { tier: Tier; className?: string }) {
  const { icon: Icon, className: tone } = TIER[tier]
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs font-semibold tracking-wide', tone, className)}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />{tier}
    </span>
  )
}

const LEVEL = {
  ok: { icon: CircleCheck, label: 'OK', className: 'bg-allow-soft text-allow-ink border-allow/30' },
  warning: { icon: TriangleAlert, label: 'Warning', className: 'bg-review-soft text-review-ink border-review/40' },
  alert: { icon: CircleAlert, label: 'Alert', className: 'bg-block-soft text-block-ink border-block/30' },
  unknown: { icon: CircleAlert, label: 'Unknown', className: 'bg-surface-2 text-secondary' },
  no_reports: { icon: CircleAlert, label: 'No reports', className: 'bg-surface-2 text-secondary' },
} as const

export function LevelBadge({ level, label }: { level: Level | 'no_reports'; label?: string }) {
  const { icon: Icon, label: fallback, className } = LEVEL[level]
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs font-semibold', className)}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />{label ?? fallback}
    </span>
  )
}

const STATUS_TONE: Record<string, string> = {
  open: 'bg-surface-2 text-secondary',
  approved: 'bg-allow-soft text-allow-ink border-allow/30',
  declined: 'bg-block-soft text-block-ink border-block/30',
  escalated: 'bg-review-soft text-review-ink border-review/40',
}
export const StatusBadge = ({ status }: { status: string }) => (
  <span className={cn('inline-flex rounded-md border px-1.5 py-0.5 text-xs font-medium capitalize', STATUS_TONE[status] ?? 'bg-surface-2')}>{status}</span>
)

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-[13px] text-secondary">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Kpi({ label, value, hint, children }: { label: string; value: ReactNode; hint?: ReactNode; children?: ReactNode }) {
  return (
    <Card className="p-4">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div className="tabular mt-1 text-2xl font-semibold tracking-tight">{value}</div>
      {hint && <div className="mt-1 text-xs text-secondary">{hint}</div>}
      {children}
    </Card>
  )
}

export function LoadingState({ rows = 4, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn('space-y-2', className)} role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} className="h-9 w-full" />)}
    </div>
  )
}

export function EmptyState({ title, description, action }: { title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <Inbox className="h-7 w-7 text-muted" aria-hidden="true" />
      <div className="text-sm font-medium">{title}</div>
      {description && <div className="max-w-md text-[13px] text-secondary">{description}</div>}
      {action}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : 'Something went wrong.'
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center" role="alert">
      <AlertTriangle className="h-7 w-7 text-block" aria-hidden="true" />
      <div className="text-sm font-medium">Could not load this</div>
      <div className="max-w-md text-[13px] text-secondary">{message}</div>
      {onRetry && <Button size="sm" onClick={onRetry}><RefreshCw className="h-3.5 w-3.5" />Try again</Button>}
    </div>
  )
}

// One wrapper so every data view has the same three states.
export function Async<T>({ query, children, rows, empty }: {
  query: { data: T | undefined; isPending: boolean; error: unknown; refetch: () => void }
  children: (data: T) => ReactNode; rows?: number; empty?: (data: T) => boolean
}) {
  if (query.isPending) return <LoadingState rows={rows} className="p-4" />
  if (query.error || query.data === undefined) return <ErrorState error={query.error} onRetry={query.refetch} />
  if (empty?.(query.data)) return null
  return <>{children(query.data)}</>
}

export function ChartTooltip({ active, label, payload, formatter, labelFormatter }: {
  active?: boolean; label?: string | number
  payload?: { name: string; value: number; color?: string; payload?: Record<string, unknown> }[]
  formatter?: (value: number, name: string) => string; labelFormatter?: (label: string | number) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-md border bg-surface px-2.5 py-2 text-xs shadow-md">
      {label !== undefined && <div className="mb-1 font-medium">{labelFormatter ? labelFormatter(label) : label}</div>}
      {payload.map((item) => (
        <div key={item.name} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-sm" style={{ background: item.color }} />
          <span className="text-secondary">{item.name}</span>
          <span className="tabular ml-auto pl-3 font-medium">{formatter ? formatter(item.value, item.name) : item.value}</span>
        </div>
      ))}
    </div>
  )
}

export const Legend = ({ items }: { items: { label: string; color: string }[] }) => (
  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-secondary">
    {items.map((item) => (
      <span key={item.label} className="inline-flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-sm" style={{ background: item.color }} />{item.label}
      </span>
    ))}
  </div>
)

export const Note = ({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'warning' }) => (
  <div className={cn('rounded-md border px-3 py-2 text-[13px]',
    tone === 'warning' ? 'border-review/40 bg-review-soft text-review-ink' : 'bg-surface-2 text-secondary')}>{children}</div>
)
