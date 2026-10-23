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
  errors: Record<string, string>
  disabled?: boolean
  advancedOpen?: boolean
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
