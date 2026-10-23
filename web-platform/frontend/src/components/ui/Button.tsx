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
  icon?: IconName
  iconPosition?: 'start' | 'end'
  loading?: boolean
  loadingLabel?: string
  fullWidth?: boolean
  children?: ReactNode
}

type ButtonElementProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'>

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
