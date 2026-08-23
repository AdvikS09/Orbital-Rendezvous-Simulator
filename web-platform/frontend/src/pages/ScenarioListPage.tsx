/**
 * Scenario selector, grouped by category.
 *
 * Shows every scenario the API offers, with the track it belongs to. Nothing is
 * hidden by track here: a professional user browsing the guided scenarios is a
 * legitimate thing to do, and the Streamlit version allowed it too.
 */

import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { ApiError, api } from '@/api/client'
import type { ScenarioSummary } from '@/api/types'
import { Callout } from '@/components/ui/Callout'
import { Icon } from '@/components/ui/Icon'
import { Spinner } from '@/components/ui/Spinner'

import './ScenarioListPage.css'

const TRACK_LABEL: Record<string, string> = {
  guided: 'Guided',
  professional: 'Professional',
  free: 'Free',
}

export function ScenarioListPage() {
  const [scenarios, setScenarios] = useState<ScenarioSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    api
      .listScenarios(undefined, controller.signal)
      .then(setScenarios)
      .catch((cause: unknown) => {
        if ((cause as Error)?.name === 'AbortError') return
        setError(
          cause instanceof ApiError
            ? cause.message
            : 'The scenario list could not be loaded.',
        )
      })
    return () => controller.abort()
  }, [])

  const grouped = useMemo(() => {
    const groups = new Map<string, { label: string; items: ScenarioSummary[] }>()
    for (const scenario of scenarios ?? []) {
      const existing = groups.get(scenario.category)
      if (existing) existing.items.push(scenario)
      else
        groups.set(scenario.category, {
          label: scenario.category_label,
          items: [scenario],
        })
    }
    return [...groups.entries()]
  }, [scenarios])

  if (error) {
    return (
      <Callout severity="error" title="Could not load scenarios">
        {error}
      </Callout>
    )
  }

  if (scenarios === null) {
    return (
      <div className="scenario-list__loading">
        <Spinner label="Loading scenarios" />
      </div>
    )
  }

  return (
    <div className="scenario-list">
      <header className="scenario-list__header">
        <h1>Scenarios</h1>
        <p>
          Every scenario runs the same validated Python engine. Each one carries
          its own assumptions and limitations, shown alongside its results.
        </p>
      </header>

      {grouped.map(([category, group]) => (
        <section key={category} className="scenario-list__group">
          <h2 className="scenario-list__group-title">{group.label}</h2>
          <ul className="scenario-list__grid" role="list">
            {group.items.map((scenario) => (
              <li key={scenario.id}>
                <Link
                  to={`/scenarios/${scenario.id}`}
                  className="scenario-card"
                  aria-label={`Open scenario: ${scenario.title}`}
                >
                  <span className="scenario-card__title">
                    {scenario.title}
                    <Icon name="chevronRight" size={16} aria-hidden />
                  </span>
                  <span className="scenario-card__summary">{scenario.summary}</span>
                  <span className="scenario-card__meta">
                    <span className="scenario-card__param-count">
                      {scenario.param_count} parameter
                      {scenario.param_count === 1 ? '' : 's'}
                    </span>
                    <span className="scenario-card__tracks">
                      {scenario.levels
                        .map((level) => TRACK_LABEL[level] ?? level)
                        .join(' · ')}
                    </span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
