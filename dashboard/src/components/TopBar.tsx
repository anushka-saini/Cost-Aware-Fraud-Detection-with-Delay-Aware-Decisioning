import { useQuery } from '@tanstack/react-query'
import { Bell, Gauge, Moon, Presentation, Search, Sun } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Overview, type Role } from '@/lib/api'
import { useTheme } from '@/lib/auth'
import { usePresentation } from '@/lib/presentation'
import { cn, sentence } from '@/lib/utils'
import { caseSearchPath } from './CommandPalette'
import { LevelBadge } from './common'
import { Button, Input, Kbd } from './ui'

type Alert = Overview['drift_alerts'][number]

// Short names for the monitoring checks, e.g. "psi" -> "PSI".
function alertType(type: string) {
  if (type === 'psi') return 'PSI'
  return sentence(type.replace(/_/g, ' '))
}

// DEMO on the public demo, LOCAL on a laptop, otherwise the environment name.
function environmentLabel(environment: string, demo: boolean) {
  if (demo) return 'DEMO'
  if (environment === 'development') return 'LOCAL'
  return environment.toUpperCase()
}

function Alerts({ alerts, role }: { alerts: Alert[]; role: Role }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  // Close when clicking anywhere else or pressing Escape.
  useEffect(() => {
    if (!open) return
    const onClick = (event: MouseEvent) => { if (!box.current?.contains(event.target as Node)) setOpen(false) }
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey) }
  }, [open])

  return (
    <div className="relative" ref={box}>
      <Button variant="ghost" size="icon" onClick={() => setOpen(!open)} aria-expanded={open}
        aria-label={alerts.length ? `${alerts.length} alerts` : 'No alerts'}>
        <Bell className="h-4 w-4" />
        {alerts.length > 0 && (
          <span className="tabular absolute -top-0.5 -right-0.5 rounded-full bg-block px-1 text-[10px] leading-4 font-semibold text-white">{alerts.length}</span>
        )}
      </Button>
      {open && (
        <div className="absolute right-0 z-30 mt-1 w-80 rounded-md border bg-surface shadow-lg">
          <div className="border-b px-3 py-2 text-xs font-medium text-muted">Monitoring alerts (warning or worse)</div>
          {alerts.length === 0 ? (
            <p className="px-3 py-4 text-[13px] text-secondary">No drift, novelty or rate alerts.</p>
          ) : (
            <ul className="max-h-72 overflow-y-auto">
              {alerts.map((alert, i) => (
                <li key={i} className="flex items-start justify-between gap-2 border-b px-3 py-2 text-[13px] last:border-b-0">
                  <div className="min-w-0">
                    <div className="truncate font-medium">{alert.name}</div>
                    <div className="text-xs text-muted">{alertType(alert.type)} · {alert.period.replace(/_/g, ' ')}</div>
                  </div>
                  <LevelBadge level={alert.level} />
                </li>
              ))}
            </ul>
          )}
          {role === 'admin' && (
            <button type="button" onClick={() => { setOpen(false); navigate('/monitoring') }}
              className="w-full border-t px-3 py-2 text-left text-[13px] text-accent hover:bg-surface-2">Open monitoring</button>
          )}
        </div>
      )}
    </div>
  )
}

export default function TopBar({ role }: { role: Role }) {
  const navigate = useNavigate()
  const { dark, toggle } = useTheme()
  const presentation = usePresentation()
  const [text, setText] = useState('')
  const status = useQuery({ queryKey: ['status'], queryFn: api.status, refetchInterval: 30_000 })
  const overview = useQuery({ queryKey: ['overview'], queryFn: api.overview, refetchInterval: 15_000 })
  const alerts = (overview.data?.drift_alerts ?? []).filter((alert) => alert.level === 'warning' || alert.level === 'alert')

  const search = (event: FormEvent) => {
    event.preventDefault()
    if (!text.trim()) return
    navigate(caseSearchPath(text))
    setText('')
  }

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-surface/95 px-6 backdrop-blur">
      <form onSubmit={search} className="relative w-full max-w-sm" role="search">
        <Search className="pointer-events-none absolute top-2.5 left-2.5 h-4 w-4 text-muted" />
        <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Case number, account or assessment id"
          aria-label="Search cases" className="pr-16 pl-8" />
        <button type="button" onClick={() => window.dispatchEvent(new Event('fraud:open-palette'))}
          className="absolute top-1.5 right-2" aria-label="Open command palette"><Kbd>Ctrl K</Kbd></button>
      </form>

      <div className="ml-auto flex items-center gap-2">
        {status.data && (
          <>
            <span className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs" title="Active model version">
              <Gauge className="h-3.5 w-3.5 text-muted" aria-hidden="true" />
              <span className="text-muted">Model</span>
              <span className="tabular font-semibold">{status.data.model.version}</span>
            </span>
            <span className={cn('rounded-md border px-2 py-0.5 text-xs font-semibold tracking-wide',
              status.data.demo_mode ? 'border-accent/40 text-accent' : 'text-secondary')} title="Where this API is running">
              {environmentLabel(status.data.environment, status.data.demo_mode)}
            </span>
          </>
        )}
        <Alerts alerts={alerts} role={role} />
        <Button variant="ghost" size="icon" onClick={presentation.toggle} aria-pressed={presentation.on}
          aria-label={presentation.on ? 'Turn off presentation mode' : 'Turn on presentation mode'}
          className={cn(presentation.on && 'bg-surface-2 text-accent')}>
          <Presentation className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={toggle} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  )
}
