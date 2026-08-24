/**
 * UI primitives.
 *
 * Presentation only: none of these know anything about orbits, the API or the
 * application's state. Import from here rather than from the individual files,
 * so a component can be moved without touching its callers.
 */

export { Icon, SEVERITY_ICON } from './Icon'
export type { IconName } from './Icon'

export { Panel } from './Panel'
export type { PanelProps } from './Panel'

export { Callout } from './Callout'
export type { CalloutProps, Severity } from './Callout'

export { Button } from './Button'
export type { ButtonProps, ButtonSize, ButtonVariant } from './Button'

export { Badge } from './Badge'
export type { BadgeProps, BadgeTone } from './Badge'

export { TrackChip, TrackChips } from './TrackChip'
export type { TrackChipProps, TrackChipsProps } from './TrackChip'

export { Disclosure } from './Disclosure'
export type { DisclosureProps } from './Disclosure'

export { Tabs, TabPanel, tabId, tabPanelId } from './Tabs'
export type { TabItem, TabPanelProps, TabsProps } from './Tabs'

export { Spinner } from './Spinner'
export type { SpinnerProps, SpinnerSize } from './Spinner'

export { EmptyState } from './EmptyState'
export type { EmptyStateProps } from './EmptyState'

// NumberValue lives one level up because it is not a primitive:
// it depends on the formatting utility. Re-exported here so callers
// have a single import site, as this module's docstring promises.
export { NumberValue } from '@/components/NumberValue'
export type { NumberValueProps } from '@/components/NumberValue'
