import { CheckCircle2, CircleAlert, X } from 'lucide-react'
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { cn } from './utils'

// Small confirmations after an action ("Case approved"). They disappear on their own.

type Tone = 'success' | 'error'
interface Toast { id: number; message: string; tone: Tone }

const ToastContext = createContext<((message: string, tone?: Tone) => void) | null>(null)
const SHOW_FOR_MS = 4000
let nextId = 1

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const dismiss = useCallback((id: number) => setToasts((all) => all.filter((t) => t.id !== id)), [])

  const show = useCallback((message: string, tone: Tone = 'success') => {
    const id = nextId++
    setToasts((all) => [...all.slice(-3), { id, message, tone }])
    window.setTimeout(() => dismiss(id), SHOW_FOR_MS)
  }, [dismiss])

  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-[60] flex w-80 flex-col gap-2" role="status" aria-live="polite">
        {toasts.map((toast) => {
          const Icon = toast.tone === 'success' ? CheckCircle2 : CircleAlert
          return (
            <div key={toast.id}
              className={cn('pointer-events-auto flex items-start gap-2 rounded-md border bg-surface px-3 py-2.5 text-[13px] shadow-lg',
                toast.tone === 'error' && 'border-block/40')}>
              <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', toast.tone === 'success' ? 'text-allow-ink' : 'text-block-ink')} aria-hidden="true" />
              <span className="flex-1">{toast.message}</span>
              <button type="button" onClick={() => dismiss(toast.id)} className="text-muted hover:text-text" aria-label="Dismiss">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const show = useContext(ToastContext)
  if (!show) throw new Error('useToast must be used inside ToastProvider')
  return show
}
