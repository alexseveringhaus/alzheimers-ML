import { useEffect, useState } from 'react'

import { toFormValues, type FormValues, type Modality } from './inputs'
import { defaultInputs } from './model/logreg'
import { PARAMS } from './model/params'
import { ApiTab } from './tabs/ApiTab'
import { PredictTab } from './tabs/PredictTab'
import { ResearchTab } from './tabs/ResearchTab'
import { ThemeContext, useThemeState, type Theme } from './theme'
import { useServerStatus } from './useServerStatus'

const TABS = [
  { id: 'predict', label: 'Predict' },
  { id: 'api', label: 'Live API' },
  { id: 'research', label: 'Research' },
] as const

type TabId = (typeof TABS)[number]['id']

const GITHUB_URL = 'https://github.com/alexseveringhaus/alzheimers-ML'

// The active tab lives in the URL hash (#predict, #api, #research), so links can open a tab
function readHash(): TabId {
  const hash = window.location.hash.slice(1)
  return TABS.find((t) => t.id === hash)?.id ?? 'predict'
}

const initialValues = () => toFormValues(defaultInputs(PARAMS))

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button type="button" className="theme-toggle" onClick={onToggle} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      {theme === 'dark' ? (
        // Sun: shown in dark mode, switches to light
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" />
          <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" />
        </svg>
      ) : (
        // Moon: shown in light mode, switches to dark
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true">
          <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z" />
        </svg>
      )}
    </button>
  )
}

export default function App() {
  const [tab, setTab] = useState<TabId>(readHash)
  const [values, setValues] = useState<FormValues>(initialValues)
  const { theme, toggle } = useThemeState()
  const server = useServerStatus()  // starts the /health call on page load to wake the API

  useEffect(() => {
    const onHashChange = () => setTab(readHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const onChange = (modality: Modality, roi: string, value: string) =>
    setValues((prev) => ({ ...prev, [modality]: { ...prev[modality], [roi]: value } }))
  const onReset = () => setValues(initialValues())

  return (
    <ThemeContext.Provider value={theme}>
      <div className="page">
        <header className="site-header">
          <a className="wordmark" href="#predict">Alzheimer's ML</a>
          <nav className="tabs" role="tablist" aria-label="Sections">
            {TABS.map((t) => (
              <a
                key={t.id}
                href={`#${t.id}`}
                role="tab"
                id={`tab-${t.id}`}
                aria-selected={tab === t.id}
                aria-controls={`panel-${t.id}`}
              >
                {t.label}
              </a>
            ))}
          </nav>
          <ThemeToggle theme={theme} onToggle={toggle} />
        </header>

        <div className="site-title">
          <h1>Alzheimer's Disease Classification</h1>
          <p>Classifying Alzheimer's disease (AD) vs cognitively normal (CN), from amyloid and tau PET scans of ADNI subjects</p>
        </div>

        {/* Inactive tabs stay mounted, so the Live API tab keeps its last response */}
        <main>
          <div role="tabpanel" id="panel-predict" aria-labelledby="tab-predict" hidden={tab !== 'predict'}>
            <PredictTab values={values} onChange={onChange} onReset={onReset} />
          </div>
          <div role="tabpanel" id="panel-api" aria-labelledby="tab-api" hidden={tab !== 'api'}>
            <ApiTab values={values} server={server} />
          </div>
          <div role="tabpanel" id="panel-research" aria-labelledby="tab-research" hidden={tab !== 'research'}>
            <ResearchTab />
          </div>
        </main>

        <footer className="site-footer">
          <p>Research tool trained on ADNI data. Not validated for clinical use; don't use it for medical decisions.</p>
          <p>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
            {' and '}
            <a href="https://adni.loni.usc.edu/" target="_blank" rel="noreferrer">ADNI data</a>
          </p>
        </footer>
      </div>
    </ThemeContext.Provider>
  )
}
