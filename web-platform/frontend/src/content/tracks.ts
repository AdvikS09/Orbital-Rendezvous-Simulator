import type { IconName } from '@/components/ui/Icon'

export type TrackId = 'guided' | 'professional' | 'free'

export interface TrackMeta {
  id: TrackId
  label: string
  description: string
  icon: IconName
}

export const TRACK_META: Record<TrackId, TrackMeta> = {
  guided: {
    id: 'guided',
    label: 'Guided',
    description: 'A step-by-step activity that asks you to predict before each run.',
    icon: 'book',
  },
  professional: {
    id: 'professional',
    label: 'Professional',
    description: 'Every parameter exposed, for open-ended, research-style work.',
    icon: 'settings',
  },
  free: {
    id: 'free',
    label: 'Free exploration',
    description: 'An open sandbox — adjust anything and run, with no set path.',
    icon: 'flask',
  },
}

export function trackMeta(level: string): TrackMeta {
  return (
    TRACK_META[level as TrackId] ?? {
      id: 'free',
      label: level,
      description: '',
      icon: 'flask',
    }
  )
}
