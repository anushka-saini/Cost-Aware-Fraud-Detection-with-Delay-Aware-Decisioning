import * as DialogPrimitive from '@radix-ui/react-dialog'
import { FileSearch, Hash, Search } from 'lucide-react'
import { useEffect, useMemo, useState, type ComponentType } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Role } from '@/lib/api'
import { pagesFor } from '@/lib/nav'
import { cn } from '@/lib/utils'
import { Kbd } from './ui'

// Ctrl+K (Cmd+K on a Mac) opens a box to jump to a page, open a case by number,
// or search cases by account or assessment id.

interface Command { id: string; label: string; hint: string; icon: ComponentType<{ className?: string }>; go: string }

export function caseSearchPath(text: string) {
  const value = text.trim().replace(/^#/, '')
  if (/^\d+$/.test(value)) return `/cases/${value}`
  return `/queue?q=${encodeURIComponent(value)}`
}

export default function CommandPalette({ role }: { role: Role }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [active, setActive] = useState(0)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setOpen((current) => !current)
      }
    }
    const onOpen = () => setOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('fraud:open-palette', onOpen)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('fraud:open-palette', onOpen)
    }
  }, [])

  useEffect(() => { if (open) { setText(''); setActive(0) } }, [open])

  const commands = useMemo(() => {
    const value = text.trim().replace(/^#/, '')
    const list: Command[] = []
    if (/^\d+$/.test(value)) list.push({ id: 'case', label: `Open case #${value}`, hint: 'Case', icon: Hash, go: `/cases/${value}` })
    else if (value) list.push({ id: 'search', label: `Search cases for "${value}"`, hint: 'Account or assessment id', icon: FileSearch, go: caseSearchPath(value) })
    for (const page of pagesFor(role)) {
      if (!value || page.label.toLowerCase().includes(value.toLowerCase())) {
        list.push({ id: page.to, label: page.label, hint: 'Page', icon: page.icon, go: page.to })
      }
    }
    return list
  }, [text, role])

  useEffect(() => setActive(0), [text])

  const run = (command: Command | undefined) => {
    if (!command) return
    setOpen(false)
    navigate(command.go)
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => Math.min(i + 1, commands.length - 1)) }
    if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => Math.max(i - 1, 0)) }
    if (event.key === 'Enter') { event.preventDefault(); run(commands[active]) }
  }

  return (
    <DialogPrimitive.Root open={open} onOpenChange={setOpen}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <DialogPrimitive.Content aria-describedby={undefined}
          className="fixed top-[15vh] left-1/2 z-50 w-[min(92vw,34rem)] -translate-x-1/2 overflow-hidden rounded-lg border bg-surface shadow-lg">
          <DialogPrimitive.Title className="sr-only">Go to</DialogPrimitive.Title>
          <div className="flex items-center gap-2 border-b px-3">
            <Search className="h-4 w-4 text-muted" aria-hidden="true" />
            <input autoFocus value={text} onChange={(e) => setText(e.target.value)} onKeyDown={onKeyDown}
              placeholder="Type a page, a case number or an account id"
              aria-label="Go to" role="combobox" aria-expanded="true" aria-controls="palette-list"
              aria-activedescendant={commands[active] ? `palette-${commands[active].id}` : undefined}
              className="h-11 flex-1 bg-transparent text-sm outline-none placeholder:text-muted" />
            <Kbd>Esc</Kbd>
          </div>
          <ul id="palette-list" role="listbox" className="max-h-80 overflow-y-auto p-1">
            {commands.length === 0 && <li className="px-3 py-6 text-center text-[13px] text-muted">Nothing matches.</li>}
            {commands.map((command, i) => {
              const Icon = command.icon
              return (
                <li key={command.id} id={`palette-${command.id}`} role="option" aria-selected={i === active}
                  onMouseEnter={() => setActive(i)} onClick={() => run(command)}
                  className={cn('flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-[13px]', i === active ? 'bg-surface-2' : '')}>
                  <Icon className="h-4 w-4 text-muted" />
                  <span className="flex-1">{command.label}</span>
                  <span className="text-xs text-muted">{command.hint}</span>
                </li>
              )
            })}
          </ul>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
