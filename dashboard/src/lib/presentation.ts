import { useCallback, useState } from 'react'

// Presentation mode makes everything bigger for a projector. It only adds a class to <html>;
// the sizes live in index.css. The choice is remembered like the theme.

export function usePresentation() {
  const [on, setOn] = useState(() => document.documentElement.classList.contains('presentation'))
  const toggle = useCallback(() => {
    setOn((current) => {
      const next = !current
      document.documentElement.classList.toggle('presentation', next)
      try { localStorage.setItem('presentation', next ? 'on' : 'off') } catch { /* private mode */ }
      return next
    })
  }, [])
  return { on, toggle }
}
