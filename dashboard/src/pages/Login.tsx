import { useQuery } from '@tanstack/react-query'
import { ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Note } from '@/components/common'
import { Button, Card, Input, Label } from '@/components/ui'
import { api } from '@/lib/api'
import { useAuth } from '@/lib/auth'

// What each role can do, shown under the demo accounts.
const ROLE_NOTES = {
  analyst: { title: 'Analyst', can: 'works the review queue' },
  admin: { title: 'Admin', can: 'also sees monitoring, model comparison and can propose thresholds' },
}

export default function Login() {
  const { signIn } = useAuth()
  // The API lists the demo accounts only when it runs in demo mode.
  const status = useQuery({ queryKey: ['status'], queryFn: api.status, retry: false })
  const demoAccounts = status.data?.demo_accounts ?? []
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
        {demoAccounts.length > 0 && (
          <div className="mt-4 space-y-2">
            <Note>Demo accounts. Click one to fill the form.</Note>
            {demoAccounts.map((account) => (
              <button key={account.username} type="button"
                onClick={() => { setUsername(account.username); setPassword(account.password) }}
                className="block w-full rounded-md border bg-surface px-3 py-2 text-left text-[13px] hover:bg-surface-2">
                <span className="font-medium">{ROLE_NOTES[account.role].title}</span>
                <span className="text-muted"> · {account.username} / {account.password}</span>
                <div className="text-xs text-secondary">{ROLE_NOTES[account.role].can}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
