import { useQuery } from '@tanstack/react-query'
import { Activity, ClipboardList, Gauge, LayoutDashboard, LogOut, Moon, Radio, ScrollText, ShieldCheck, SlidersHorizontal, Sun, Trophy } from 'lucide-react'
import type { ComponentType } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { LevelBadge } from './components/common'
import { Button, Skeleton } from './components/ui'
import { api, type Role } from './lib/api'
import { useAuth, useTheme } from './lib/auth'
import { cn } from './lib/utils'
import AuditLog from './pages/AuditLog'
import CaseDetail from './pages/CaseDetail'
import LiveFeed from './pages/LiveFeed'
import Login from './pages/Login'
import ModelComparison from './pages/ModelComparison'
import Monitoring from './pages/Monitoring'
import Overview from './pages/Overview'
import ReviewQueue from './pages/ReviewQueue'
import Simulator from './pages/Simulator'

interface NavItem { to: string; label: string; icon: ComponentType<{ className?: string }>; roles: Role[] }

// Analysts work transactions; admins also own the model and its thresholds.
const NAV: { heading: string; items: NavItem[] }[] = [
  { heading: 'Operations', items: [
    { to: '/', label: 'Overview', icon: LayoutDashboard, roles: ['analyst', 'admin'] },
    { to: '/live', label: 'Live feed', icon: Radio, roles: ['analyst', 'admin'] },
    { to: '/queue', label: 'Review queue', icon: ClipboardList, roles: ['analyst', 'admin'] },
  ] },
  { heading: 'Model', items: [
    { to: '/simulator', label: 'Threshold simulator', icon: SlidersHorizontal, roles: ['analyst', 'admin'] },
    { to: '/monitoring', label: 'Monitoring', icon: Activity, roles: ['admin'] },
    { to: '/models', label: 'Model comparison', icon: Trophy, roles: ['admin'] },
  ] },
  { heading: 'Governance', items: [
    { to: '/audit', label: 'Audit log', icon: ScrollText, roles: ['analyst', 'admin'] },
  ] },
]

function Shell() {
  const { user, signOut } = useAuth()
  const { dark, toggle } = useTheme()
  const overview = useQuery({ queryKey: ['overview'], queryFn: api.overview, refetchInterval: 15_000 })
  const kpis = overview.data?.kpis
  const role = user!.role

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-56 shrink-0 flex-col border-r bg-surface">
        <div className="flex items-center gap-2 px-4 py-4">
          <ShieldCheck className="h-5 w-5 text-accent" aria-hidden="true" />
          <div className="leading-tight">
            <div className="text-sm font-semibold">Fraud Operations</div>
            <div className="text-[11px] text-muted">Cost-aware decisioning</div>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto px-2" aria-label="Main">
          {NAV.map((group) => {
            const items = group.items.filter((item) => item.roles.includes(role))
            if (!items.length) return null
            return (
              <div key={group.heading} className="mb-3">
                <div className="px-2 pb-1 text-[11px] font-medium tracking-wide text-muted uppercase">{group.heading}</div>
                {items.map(({ to, label, icon: Icon }) => (
                  <NavLink key={to} to={to} end={to === '/'}
                    className={({ isActive }) => cn('flex items-center gap-2 rounded-md px-2 py-1.5 text-[13px]',
                      isActive ? 'bg-surface-2 font-medium text-text' : 'text-secondary hover:bg-surface-2')}>
                    <Icon className="h-4 w-4" />
                    <span className="flex-1">{label}</span>
                    {to === '/queue' && kpis && kpis.review_queue > 0 && (
                      <span className="tabular rounded bg-review-soft px-1.5 text-[11px] font-semibold text-review-ink">{kpis.review_queue}</span>
                    )}
                  </NavLink>
                ))}
              </div>
            )
          })}
        </nav>
        <div className="space-y-2 border-t p-3 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1 text-muted"><Gauge className="h-3.5 w-3.5" />Model</span>
            {kpis ? <span className="tabular font-medium">{kpis.active_model.version}</span> : <Skeleton className="h-4 w-8" />}
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted">Drift</span>
            {kpis ? <LevelBadge level={kpis.drift_status} /> : <Skeleton className="h-4 w-14" />}
          </div>
          <div className="flex items-center justify-between gap-2 pt-1">
            <div className="min-w-0">
              <div className="truncate font-medium">{user!.username}</div>
              <div className="text-muted capitalize">{role}</div>
            </div>
            <div className="flex gap-1">
              <Button variant="ghost" size="icon" onClick={toggle} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
                {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="icon" onClick={signOut} aria-label="Sign out"><LogOut className="h-4 w-4" /></Button>
            </div>
          </div>
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-6 py-5">
        <div className="mx-auto max-w-[1400px]">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/live" element={<LiveFeed />} />
            <Route path="/queue" element={<ReviewQueue />} />
            <Route path="/cases/:id" element={<CaseDetail />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/audit" element={<AuditLog />} />
            {role === 'admin' && <Route path="/monitoring" element={<Monitoring />} />}
            {role === 'admin' && <Route path="/models" element={<ModelComparison />} />}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  const { user, loading } = useAuth()
  const location = useLocation()
  if (loading) return <div className="p-8"><Skeleton className="h-8 w-64" /></div>
  if (!user) return <Login />
  if (location.pathname === '/login') return <Navigate to="/" replace />
  return <Shell />
}
