import { forwardRef } from 'react'
import type {
  ButtonHTMLAttributes,
  ForwardRefExoticComponent,
  ReactNode,
  RefAttributes,
} from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import { Spinner } from './Spinner'
import './Button.css'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ButtonSize = 'sm' | 'md'

interface ButtonOwnProps {
  variant?: ButtonVariant
  size?: ButtonSize
  /** Leading glyph. Purely supporting: the label still carries the meaning. */
  icon?: IconName
  /** Place the icon after the label instead of before it. */
  iconPosition?: 'start' | 'end'
  /** Replaces the icon with a spinner, disables the control, sets aria-busy. */
  loading?: boolean
  /** Announced while loading. Defaults to 'Working'. */
  loadingLabel?: string
  fullWidth?: boolean
  children?: ReactNode
}

type ButtonElementProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'>

/**
 * A button with no visible label must carry an explicit accessible name.
 * The union makes that a compile error rather than an audit finding.
 */
export type ButtonProps = ButtonOwnProps &
  ButtonElementProps &
  ({ children: ReactNode } | { children?: undefined; 'aria-label': string })

type ButtonImplProps = ButtonOwnProps & ButtonElementProps

const ButtonImpl = forwardRef<HTMLButtonElement, ButtonImplProps>(function Button(
  {
    variant = 'secondary',
    size = 'md',
    icon,
    iconPosition = 'start',
    loading = false,
    loadingLabel = 'Working',
    fullWidth = false,
    className,
    children,
    disabled,
    type = 'button',
    ...rest
  },
  ref,
) {
  const hasLabel = children !== undefined && children !== null && children !== false
  const iconSize = size === 'sm' ? 13 : 15

  const classes = [
    'button',
    `button--${variant}`,
    `button--${size}`,
    fullWidth ? 'button--full' : '',
    hasLabel ? '' : 'button--icon-only',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  // While loading the spinner occupies the icon slot, so the control keeps its
  // width and the layout around it does not shift.
  const glyph = loading ? (
    <Spinner size="sm" inherit label={loadingLabel} />
  ) : icon ? (
    <Icon name={icon} size={iconSize} className="button__icon" />
  ) : null

  return (
    <button
      {...rest}
      ref={ref}
      type={type}
      className={classes}
      disabled={disabled === true || loading}
      aria-busy={loading || undefined}
    >
      {iconPosition === 'start' ? glyph : null}
      {hasLabel ? <span className="button__label">{children}</span> : null}
      {iconPosition === 'end' ? glyph : null}
    </button>
  )
})

export const Button = ButtonImpl as ForwardRefExoticComponent<
  ButtonProps & RefAttributes<HTMLButtonElement>
>
