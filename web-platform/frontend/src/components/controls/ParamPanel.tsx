/**
 * The parameter form for a scenario.
 *
 * Ordinary parameters sit in a responsive grid; anything the backend marks
 * advanced is folded into a disclosure, closed by default. That split is the
 * whole reason the panel exists: a scenario can expose twenty controls, and a
 * student needs the four that matter without the other sixteen shouting at
 * them, while an engineer needs all twenty one click away.
 *
 * The panel holds no parameter state. Values come in, changes go out by name,
 * and validation messages come from the server, so the form can never disagree
 * with what the engine will actually accept.
 */

import { useMemo } from 'react'
import type { ParamSpec, ParamValues } from '@/api/types'
import { Disclosure } from '@/components/ui'
import { ParamControl } from './ParamControl'
import type { ParamValue } from './ParamControl'
import './ParamPanel.css'

export interface ParamPanelProps {
  params: ParamSpec[]
  values: ParamValues
  onChange: (name: string, value: ParamValue) => void
  /** Field-level messages keyed by parameter name. */
  errors: Record<string, string>
  disabled?: boolean
  /** Open the advanced section on first render. */
  advancedOpen?: boolean
  /** Accessible name for the form region. */
  legend?: string
  className?: string
}

export function ParamPanel({
  params,
  values,
  onChange,
  errors,
  disabled = false,
  advancedOpen = false,
  legend = 'Parameters',
  className,
}: ParamPanelProps) {
  const { basic, advanced } = useMemo(() => {
    const basicParams: ParamSpec[] = []
    const advancedParams: ParamSpec[] = []
    for (const param of params) {
      if (param.advanced) advancedParams.push(param)
      else basicParams.push(param)
    }
    return { basic: basicParams, advanced: advancedParams }
  }, [params])

  const classes = ['param-panel', className ?? ''].filter(Boolean).join(' ')

  function renderControl(param: ParamSpec) {
    return (
      <div className="param-panel__cell" key={param.name}>
        <ParamControl
          param={param}
          value={values[param.name]}
          onChange={(next) => onChange(param.name, next)}
          error={errors[param.name]}
          disabled={disabled}
        />
      </div>
    )
  }

  return (
    <div className={classes}>
      <fieldset className="param-panel__fieldset">
        <legend className="visually-hidden">{legend}</legend>
        {basic.length > 0 ? (
          <div className="param-panel__grid">{basic.map(renderControl)}</div>
        ) : null}
      </fieldset>

      {advanced.length > 0 ? (
        <Disclosure
          title="Advanced controls"
          count={advanced.length}
          defaultOpen={advancedOpen}
          icon="settings"
          className="param-panel__advanced"
        >
          <fieldset className="param-panel__fieldset">
            <legend className="visually-hidden">Advanced parameters</legend>
            <div className="param-panel__grid">{advanced.map(renderControl)}</div>
          </fieldset>
        </Disclosure>
      ) : null}
    </div>
  )
}
