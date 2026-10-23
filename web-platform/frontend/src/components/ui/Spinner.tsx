import './Spinner.css'

export type SpinnerSize = 'sm' | 'md' | 'lg'

export interface SpinnerProps {
  size?: SpinnerSize
  label?: string
  inherit?: boolean
  className?: string
}

export function Spinner({
  size = 'sm',
  label = 'Working',
  inherit = false,
  className,
}: SpinnerProps) {
  const classes = [
    'spinner',
    `spinner--${size}`,
    inherit ? 'spinner--inherit' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <span role="status" className="spinner-wrap">
      <span className={classes} aria-hidden="true" />
      <span className="visually-hidden">{label}</span>
    </span>
  )
}
