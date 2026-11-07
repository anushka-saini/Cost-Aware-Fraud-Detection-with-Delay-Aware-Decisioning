import { useQuery } from '@tanstack/react-query'
import { LogOut, ShieldCheck } from 'lucide-react'
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import CommandPalette from './components/CommandPalette'
import TopBar from './components/TopBar'
import { Button, Skeleton } from './components/ui'
import { api } from './lib/api'
import { useAuth } from './lib/auth'
import { NAV } from './lib/nav'
import { cn } from './lib/utils'
import About from './pages/About'
import AuditLog from './pages/AuditLog'
import CaseDetail from './pages/CaseDetail'
import Explainability from './pages/Explainability'
import LiveFeed from './pages/LiveFeed'
import Login from './pages/Login'
import ModelLab from './pages/ModelLab'
import Monitoring from './pages/Monitoring'
import Overview from './pages/Overview'
import ReviewQueue from './pages/ReviewQueue'
import Simulator from './pages/Simulator'
import Status from './pages/Status'

function Shell() {
  const { user, signOut } = useAuth()
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
        <div className="flex items-center justify-between gap-2 border-t p-3 text-xs">
          <div className="min-w-0">
            <div className="truncate font-medium">{user!.username}</div>
            <div className="text-muted capitalize">{role}</div>
          </div>
          <Button variant="ghost" size="icon" onClick={signOut} aria-label="Sign out"><LogOut className="h-4 w-4" /></Button>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar role={role} />
        <main className="min-w-0 flex-1 px-6 py-5">
          <div className="mx-auto max-w-[1400px]">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/live" element={<LiveFeed />} />
              <Route path="/queue" element={<ReviewQueue />} />
              <Route path="/cases/:id" element={<CaseDetail />} />
              <Route path="/simulator" element={<Simulator />} />
              <Route path="/explain" element={<Explainability />} />
              <Route path="/audit" element={<AuditLog />} />
              <Route path="/status" element={<Status />} />
              <Route path="/about" element={<About />} />
              {role === 'admin' && <Route path="/monitoring" element={<Monitoring />} />}
              {role === 'admin' && <Route path="/models" element={<ModelLab />} />}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
      <CommandPalette role={role} />
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
