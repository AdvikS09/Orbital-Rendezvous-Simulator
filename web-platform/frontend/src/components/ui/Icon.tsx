/**
 * Line icons.
 *
 * The Streamlit version used emoji as status markers and section decoration.
 * Emoji render inconsistently across platforms, carry a tone that does not suit
 * a research tool, and are announced verbatim by screen readers ("triangular
 * ruler"), which is noise rather than information.
 *
 * These replace them: restrained 1.5px stroke glyphs on a 24-unit grid,
 * inheriting `currentColor` so they take the colour of the surrounding text.
 * Every icon is `aria-hidden` by default, because in this interface an icon
 * always sits beside a text label that already says what it means. Pass a
 * `title` only where an icon genuinely stands alone.
 */

import type { SVGProps } from 'react'

export type IconName =
  | 'info'
  | 'success'
  | 'warning'
  | 'error'
  | 'orbit'
  | 'chart'
  | 'download'
  | 'play'
  | 'reset'
  | 'settings'
  | 'book'
  | 'flask'
  | 'shield'
  | 'chevronRight'
  | 'chevronDown'
  | 'chevronLeft'
  | 'external'
  | 'check'
  | 'close'
  | 'menu'
  | 'clock'
  | 'ruler'
  | 'target'
  | 'lock'
  | 'user'
  | 'table'

interface IconProps extends Omit<SVGProps<SVGSVGElement>, 'name'> {
  name: IconName
  size?: number
  title?: string
}

const PATHS: Record<IconName, JSX.Element> = {
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </>
  ),
  success: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m8.5 12.5 2.5 2.5 4.5-5" />
    </>
  ),
  warning: (
    <>
      <path d="M10.3 4.3 2.8 17.5A1.9 1.9 0 0 0 4.5 20.4h15a1.9 1.9 0 0 0 1.7-2.9L13.7 4.3a1.9 1.9 0 0 0-3.4 0Z" />
      <path d="M12 9.5v4M12 17h.01" />
    </>
  ),
  error: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m15 9-6 6M9 9l6 6" />
    </>
  ),
  orbit: (
    <>
      <circle cx="12" cy="12" r="3" />
      <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-28 12 12)" />
      <circle cx="20" cy="8.4" r="1.4" />
    </>
  ),
  chart: (
    <>
      <path d="M3 3v16.5A1.5 1.5 0 0 0 4.5 21H21" />
      <path d="m7 15 3.5-4 3 2.5L19 7" />
    </>
  ),
  download: (
    <>
      <path d="M12 3v11.5" />
      <path d="m7.5 10.5 4.5 4.5 4.5-4.5" />
      <path d="M4 18.5v1A1.5 1.5 0 0 0 5.5 21h13a1.5 1.5 0 0 0 1.5-1.5v-1" />
    </>
  ),
  play: <path d="M7 4.8v14.4a.6.6 0 0 0 .93.5l11-7.2a.6.6 0 0 0 0-1l-11-7.2a.6.6 0 0 0-.93.5Z" />,
  reset: (
    <>
      <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1" />
      <path d="M3 3.5V9h5.5" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.6 1.6 0 0 0 .32 1.77l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.6 1.6 0 0 0-1.77-.32 1.6 1.6 0 0 0-1 1.47V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.47 1.6 1.6 0 0 0-1.77.32l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.6 1.6 0 0 0 .32-1.77 1.6 1.6 0 0 0-1.47-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.47-1 1.6 1.6 0 0 0-.32-1.77l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.6 1.6 0 0 0 1.77.32H9a1.6 1.6 0 0 0 1-1.47V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.47 1.6 1.6 0 0 0 1.77-.32l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.6 1.6 0 0 0-.32 1.77V9a1.6 1.6 0 0 0 1.47 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.47 1Z" />
    </>
  ),
  book: (
    <>
      <path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H19v15H5.5A1.5 1.5 0 0 0 4 19.5Z" />
      <path d="M4 19.5A1.5 1.5 0 0 1 5.5 18H19v3H5.5A1.5 1.5 0 0 1 4 19.5Z" />
    </>
  ),
  flask: (
    <>
      <path d="M9.5 3h5M10.5 3v6.2L5.4 18a1.6 1.6 0 0 0 1.4 2.4h10.4a1.6 1.6 0 0 0 1.4-2.4l-5.1-8.8V3" />
      <path d="M7.6 14.5h8.8" />
    </>
  ),
  shield: <path d="M12 3 4.5 6v5.6c0 4.4 3 8.3 7.5 9.4 4.5-1.1 7.5-5 7.5-9.4V6Z" />,
  chevronRight: <path d="m9.5 5.5 6.5 6.5-6.5 6.5" />,
  chevronDown: <path d="m5.5 9.5 6.5 6.5 6.5-6.5" />,
  chevronLeft: <path d="M14.5 5.5 8 12l6.5 6.5" />,
  external: (
    <>
      <path d="M13.5 4.5H19.5v6" />
      <path d="M19.5 4.5 11 13" />
      <path d="M18 14.5v4a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6h4" />
    </>
  ),
  check: <path d="m5 12.5 5 5L19 6.5" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5.2l3.2 2" />
    </>
  ),
  ruler: (
    <>
      <path d="m4.2 15.4 11.2-11.2a1.4 1.4 0 0 1 2 0l2.4 2.4a1.4 1.4 0 0 1 0 2L8.6 19.8a1.4 1.4 0 0 1-2 0l-2.4-2.4a1.4 1.4 0 0 1 0-2Z" />
      <path d="m8 11.6 1.8 1.8M11 8.6l1.8 1.8M14 5.6l1.8 1.8" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  lock: (
    <>
      <rect x="4.5" y="10.5" width="15" height="10" rx="1.6" />
      <path d="M8 10.5V7.8a4 4 0 0 1 8 0v2.7" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="3.6" />
      <path d="M4.8 20.2a7.4 7.4 0 0 1 14.4 0" />
    </>
  ),
  table: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="1.6" />
      <path d="M3.5 9.5h17M9.5 9.5v10" />
    </>
  ),
}

export function Icon({ name, size = 16, title, ...rest }: IconProps) {
  const glyph = PATHS[name]
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
      {...rest}
    >
      {title ? <title>{title}</title> : null}
      {glyph}
    </svg>
  )
}

/** Icon paired with each status severity. */
export const SEVERITY_ICON = {
  info: 'info',
  success: 'success',
  warning: 'warning',
  error: 'error',
} as const satisfies Record<string, IconName>
