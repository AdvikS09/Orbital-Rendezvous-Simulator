/**
 * The three ways a scenario can be worked through.
 *
 * These are not difficulty levels and not a claim about the person using them:
 * they describe how much of the interface is exposed and how guided the path is.
 * A single scenario usually belongs to more than one — the free-drift scenario,
 * for instance, is both a guided activity and something to explore freely.
 *
 * Kept in one place so the label, the one-line meaning and the marker icon are
 * identical everywhere a track is shown (the scenario list, a scenario header).
 */

import type { IconName } from '@/components/ui/Icon'

export type TrackId = 'guided' | 'professional' | 'free'

export interface TrackMeta {
  id: TrackId
  /** Shown on the chip. */
  label: string
  /** One line, used as the chip's tooltip. States what the track is, plainly. */
  description: string
  /** A distinct glyph per track, so the three differ by shape and not by colour. */
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
    // "Professional" here means interface depth, not a credential or a claim
    // about the user: every parameter is exposed for research-style workflows.
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

/** Resolve an API level string to its display metadata, falling back gracefully. */
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
