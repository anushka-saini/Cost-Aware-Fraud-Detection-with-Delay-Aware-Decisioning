import { useEffect, useState } from 'react'
import { getHealth, getModelInfo } from './api.js'
import { MODEL_NAMES } from './format.js'
import Scoring from './views/Scoring.jsx'
import Validation from './views/Validation.jsx'
import DecisionLog from './views/DecisionLog.jsx'

const TABS = [
  { id: 'scoring', label: 'Live Scoring' },
  { id: 'validation', label: 'Model Validation' },
  { id: 'decisions', label: 'Decision Log' },
]

const tabFromHash = () => {
  const id = window.location.hash.replace('#', '')
  return TABS.some((t) => t.id === id) ? id : 'scoring'
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash)
  const [apiUp, setApiUp] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const check = () =>
      getHealth()
        .then(() => setApiUp(true))
        .catch(() => setApiUp(false))
    check()
    const timer = setInterval(check, 10000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (apiUp) getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
  }, [apiUp])

  const select = (id) => {
    window.location.hash = id
    setTab(id)
  }

  const modelName = modelInfo ? MODEL_NAMES[modelInfo.model_class] || modelInfo.model_class : null

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <h1>Cost-Aware Fraud Detection</h1>
            <p>PaySim transactions · scoring, validation and decision analytics</p>
          </div>
          <nav className="tabs" role="tablist">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                className="tab"
                aria-selected={tab === t.id}
                onClick={() => select(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="api-status">
            <i className={`dot ${apiUp === null ? '' : apiUp ? 'ok' : 'down'}`} />
            {apiUp === null && 'Checking API…'}
            {apiUp === false && 'API not reachable'}
            {apiUp && (modelName ? `API connected · model loaded: ${modelName}` : 'API connected')}
          </div>
        </div>
      </header>
      <main>
        {tab === 'scoring' && <Scoring modelInfo={modelInfo} />}
        {tab === 'validation' && <Validation modelInfo={modelInfo} />}
        {tab === 'decisions' && <DecisionLog />}
      </main>
    </>
  )
}
