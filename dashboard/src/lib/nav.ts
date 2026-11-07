import { Activity, ClipboardList, FlaskConical, Info, LayoutDashboard, Lightbulb, Radio, ScrollText, Server, SlidersHorizontal } from 'lucide-react'
import type { ComponentType } from 'react'
import type { Role } from './api'

// The pages in the sidebar. The command palette (Ctrl+K) uses the same list.

export interface NavItem { to: string; label: string; icon: ComponentType<{ className?: string }>; roles: Role[] }

// Analysts work transactions; admins also own the model and its thresholds.
export const NAV: { heading: string; items: NavItem[] }[] = [
  { heading: 'Operations', items: [
    { to: '/', label: 'Overview', icon: LayoutDashboard, roles: ['analyst', 'admin'] },
    { to: '/live', label: 'Live feed', icon: Radio, roles: ['analyst', 'admin'] },
    { to: '/queue', label: 'Review queue', icon: ClipboardList, roles: ['analyst', 'admin'] },
  ] },
  { heading: 'Model', items: [
    { to: '/simulator', label: 'Threshold simulator', icon: SlidersHorizontal, roles: ['analyst', 'admin'] },
    { to: '/explain', label: 'Explainability', icon: Lightbulb, roles: ['analyst', 'admin'] },
    { to: '/monitoring', label: 'Monitoring', icon: Activity, roles: ['admin'] },
    { to: '/models', label: 'Model lab', icon: FlaskConical, roles: ['admin'] },
  ] },
  { heading: 'Governance', items: [
    { to: '/audit', label: 'Audit log', icon: ScrollText, roles: ['analyst', 'admin'] },
    { to: '/status', label: 'System status', icon: Server, roles: ['analyst', 'admin'] },
    { to: '/about', label: 'About', icon: Info, roles: ['analyst', 'admin'] },
  ] },
]

export function pagesFor(role: Role) {
  return NAV.flatMap((group) => group.items.filter((item) => item.roles.includes(role)))
}
