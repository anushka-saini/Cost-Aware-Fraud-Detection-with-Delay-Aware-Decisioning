import { useCallback, useEffect, useState } from 'react'

// Runs an API call on mount and exposes { data, error, loading, reload }.
export default function useFetch(fetcher) {
  const [state, setState] = useState({ data: null, error: null, loading: true })

  const reload = useCallback(() => {
    setState((s) => ({ ...s, loading: true }))
    fetcher()
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => setState({ data: null, error, loading: false }))
  }, [fetcher])

  useEffect(reload, [reload])

  return { ...state, reload }
}
