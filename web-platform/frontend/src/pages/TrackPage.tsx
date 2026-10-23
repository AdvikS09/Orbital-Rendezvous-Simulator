import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '@/api/client'
import type { Track } from '@/api/types'
import { Button, Callout } from '@/components/ui'
import { useSession } from '@/state/session'
import { suggestedTrack, trackDestination } from './participantFlow'
import './TrackPage.css'

interface TrackCard {
  id: Track
  title: string
  description: string
  audience: string
}

const CARDS: readonly TrackCard[] = [
  {
    id: 'guided',
    title: 'Guided educational activities',
    description:
      'Three short activities that build from “what keeps a satellite up” to ' +
      '“how much can you trust a linear model”. Each step asks you to predict ' +
      'what will happen before you run it.',
    audience:
      'Best if you are new to orbital mechanics. About 20–25 minutes each. ' +
      'Includes optional anonymous before-and-after questions so a teacher can ' +
      'see whether the activities worked.',
  },
  {
    id: 'professional',
    title: 'Professional / research exploration',
    description:
      'Direct access to every scenario, including the model-comparison study, ' +
      'the integrator convergence check and the external-validation export. ' +
      'Full parameter control and reproducible configuration downloads.',
    audience:
      'Best if you already know the subject and want to interrogate the model.',
  },
  {
    id: 'free',
    title: 'Free simulation mode',
    description:
      'Every scenario, no structure, no assessments. Pick something and run it.',
    audience: 'Best if you want to explore on your own terms.',
  },
]

export function TrackPage() {
  const navigate = useNavigate()
  const { userType, isSignedIn, setTrack } = useSession()
  const [choosing, setChoosing] = useState<Track | null>(null)

  const recommended = suggestedTrack(userType)

  async function choose(track: Track) {
    setChoosing(track)
    setTrack(track)

    if (isSignedIn) {
      try {
        await api.selectTrack(track)
      } catch {}
    }

    navigate(trackDestination(track, false))
  }

  return (
    <div className="track-page">
      <header className="track-page__header">
        <h1>How would you like to use the simulator?</h1>
        <p className="track-page__subtitle">You can switch tracks at any time.</p>
      </header>

      {isSignedIn ? null : (
        <Callout severity="info">
          <p>
            You are continuing without a tracked session, so this choice is not
            recorded. Everything in the simulator still works.
          </p>
        </Callout>
      )}

      <ul className="track-page__cards" role="list">
        {CARDS.map((card) => {
          const isRecommended = card.id === recommended
          return (
            <li key={card.id}>
              <article
                className={
                  isRecommended
                    ? 'track-page__card track-page__card--recommended'
                    : 'track-page__card'
                }
                aria-labelledby={`track-${card.id}-title`}
              >
                <h2 className="track-page__title" id={`track-${card.id}-title`}>
                  {card.title}
                  {isRecommended ? (
                    <span className="track-page__suggested">
                      suggested for you
                    </span>
                  ) : null}
                </h2>
                <p className="track-page__description">{card.description}</p>
                <p className="track-page__audience">{card.audience}</p>
                <div className="track-page__action">
                  <Button
                    type="button"
                    variant={isRecommended ? 'primary' : 'secondary'}
                    disabled={choosing !== null}
                    onClick={() => {
                      void choose(card.id)
                    }}
                  >
                    Choose {card.title.toLowerCase()}
                  </Button>
                </div>
              </article>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
