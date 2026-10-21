import { ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Note } from '@/components/common'
import { Button, Card, Input, Label } from '@/components/ui'
import { useAuth } from '@/lib/auth'

const DEMO = [
  { role: 'Analyst', username: 'analyst', password: 'analyst123', can: 'works the review queue' },
  { role: 'Admin', username: 'admin', password: 'admin123', can: 'also sees monitoring, model comparison and can propose thresholds' },
]

export default function Login() {
  const { signIn } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await signIn(username, password)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Sign-in failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-sm">
        <div className="mb-5 flex items-center gap-2">
          <ShieldCheck className="h-6 w-6 text-accent" aria-hidden="true" />
          <div>
            <div className="text-lg font-semibold">Fraud Operations</div>
            <div className="text-xs text-muted">Cost-aware fraud detection on PaySim</div>
          </div>
        </div>
        <Card className="p-5">
          <form onSubmit={submit} className="space-y-3">
            <div>
              <Label htmlFor="username">Username</Label>
              <Input id="username" autoComplete="username" autoFocus required value={username} onChange={(e) => setUsername(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
            {error && <div role="alert" className="text-[13px] text-block-ink">{error}</div>}
            <Button type="submit" variant="primary" className="w-full" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</Button>
          </form>
        </Card>
        <div className="mt-4 space-y-2">
          <Note>Demo accounts. Click one to fill the form.</Note>
          {DEMO.map((account) => (
            <button key={account.username} type="button"
              onClick={() => { setUsername(account.username); setPassword(account.password) }}
              className="block w-full rounded-md border bg-surface px-3 py-2 text-left text-[13px] hover:bg-surface-2">
              <span className="font-medium">{account.role}</span>
              <span className="text-muted"> · {account.username} / {account.password}</span>
              <div className="text-xs text-secondary">{account.can}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
