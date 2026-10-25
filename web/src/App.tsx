import { useEffect, useState } from 'react'

import { toFormValues, type FormValues, type Modality } from './inputs'
import { defaultInputs } from './model/logreg'
import { PARAMS } from './model/params'
import { ApiTab } from './tabs/ApiTab'
import { PredictTab } from './tabs/PredictTab'
import { ResearchTab } from './tabs/ResearchTab'
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

export default function App() {
  const [tab, setTab] = useState<TabId>(readHash)
  const [values, setValues] = useState<FormValues>(initialValues)
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
    <div className="page">
      <header>
        <h1>Alzheimer's Disease Classification</h1>
        <p className="subtitle">AD vs cognitively normal, from amyloid and tau PET scans of ADNI subjects</p>
        <p className="disclaimer" role="note">
          <strong>Research tool only.</strong> This model was trained on ADNI research data and is not validated for
          clinical use. Do not use it for medical decisions.
        </p>
      </header>

      <nav className="tabs" role="tablist" aria-label="Sections">
        {TABS.map((t) => (
          <a
            key={t.id}
            href={`#${t.id}`}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            className={tab === t.id ? 'active' : undefined}
          >
            {t.label}
          </a>
        ))}
      </nav>

      {/* Inactive tabs stay mounted, so the Live API tab keeps its last response */}
      <main>
        <div role="tabpanel" id="panel-predict" aria-labelledby="tab-predict" hidden={tab !== 'predict'}>
          <PredictTab values={values} onChange={onChange} onReset={onReset} />
        </div>
        <div role="tabpanel" id="panel-api" aria-labelledby="tab-api" hidden={tab !== 'api'}>
          <ApiTab values={values} onChange={onChange} onReset={onReset} server={server} />
        </div>
        <div role="tabpanel" id="panel-research" aria-labelledby="tab-research" hidden={tab !== 'research'}>
          <ResearchTab />
        </div>
      </main>

      <footer>
        <a href={GITHUB_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
        <span aria-hidden="true">·</span>
        <span>Data: <a href="https://adni.loni.usc.edu/" target="_blank" rel="noreferrer">ADNI</a></span>
      </footer>
    </div>
  )
}
