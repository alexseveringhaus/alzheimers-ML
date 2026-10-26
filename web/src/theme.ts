import { createContext, useContext, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

// index.html reads the same key before first paint, so a saved choice never flashes
const STORAGE_KEY = 'theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'

// Storage can be blocked (private windows, strict settings); the toggle still works for the visit
function readSaved(): Theme | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY)
    return value === 'light' || value === 'dark' ? value : null
  } catch {
    return null
  }
}

function writeSaved(theme: Theme | null) {
  try {
    if (theme) window.localStorage.setItem(STORAGE_KEY, theme)
    else window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Ignore: the choice just won't be remembered
  }
}

/**
 * The page theme: the visitor's saved choice, else the system setting (followed live).
 * Toggling back to the system's theme clears the saved choice.
 */
export function useThemeState(): { theme: Theme; toggle: () => void } {
  const [saved, setSaved] = useState<Theme | null>(readSaved)
  const [systemDark, setSystemDark] = useState(() => window.matchMedia(DARK_QUERY).matches)

  useEffect(() => {
    const query = window.matchMedia(DARK_QUERY)
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const system: Theme = systemDark ? 'dark' : 'light'
  const theme = saved ?? system

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    const choice = next === system ? null : next
    setSaved(choice)
    writeSaved(choice)
  }

  return { theme, toggle }
}

export const ThemeContext = createContext<Theme>('light')

/** The current page theme, for components that need hex colors rather than CSS variables. */
export function useTheme(): Theme {
  return useContext(ThemeContext)
}
