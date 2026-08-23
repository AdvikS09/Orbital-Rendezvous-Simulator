import './Spinner.css'

export type SpinnerSize = 'sm' | 'md' | 'lg'

export interface SpinnerProps {
  size?: SpinnerSize
  /** Announced by assistive technology. Kept visually hidden. */
  label?: string
  /** Take the surrounding text colour rather than the accent. */
  inherit?: boolean
  className?: string
}

/**
 * Indeterminate activity indicator.
 *
 * The visible ring is decorative; the accessible name comes from the adjacent
 * visually-hidden text, so the state is announced rather than merely drawn.
 */
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
