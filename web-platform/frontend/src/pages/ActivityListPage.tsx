import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import type { Activity } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Badge, Callout, Disclosure, Panel, Spinner } from '@/components/ui'
import { ROUTES } from './participantFlow'
import './ActivityListPage.css'

export function ActivityListPage() {
  const [activities, setActivities] = useState<Activity[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    api
      .listActivities(controller.signal)
      .then(setActivities)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setLoadError(
          cause instanceof ApiError
            ? cause.message
            : 'The activities could not be loaded.',
        )
      })
    return () => controller.abort()
  }, [])

  return (
    <div className="activity-list">
      <header className="activity-list__header">
        <h1>Choose an activity</h1>
        <p className="activity-list__subtitle">
          Work through them in order if you are new to this — each builds on the
          one before.
        </p>
      </header>

      <Callout severity="warning" title="Requires teacher review before classroom use">
        <p>
          These activities were written against what the engine genuinely does,
          and every expected outcome quoted in them was checked against the
          simulator. They have not been reviewed by a subject teacher or trialled
          with students — reading level, pacing and curriculum fit are all
          unverified.
        </p>
      </Callout>

      {loadError !== null ? (
        <Callout severity="error" title="Activities unavailable">
          <p>{loadError}</p>
        </Callout>
      ) : null}

      {activities === null && loadError === null ? (
        <p className="activity-list__loading">
          <Spinner label="Loading the activities" />
          <span aria-hidden="true">Loading the activities…</span>
        </p>
      ) : null}

      <ul className="activity-list__items" role="list">
        {(activities ?? []).map((activity) => (
          <li key={activity.id}>
            <Panel title={activity.title} headingLevel={2} icon="flask">
              <p className="activity-list__summary">{activity.summary}</p>

              <dl className="activity-list__facts">
                <div className="activity-list__fact">
                  <dt>Level</dt>
                  <dd>{activity.level}</dd>
                </div>
                <div className="activity-list__fact">
                  <dt>Duration</dt>
                  <dd>
                    about{' '}
                    <NumberValue
                      value={activity.duration_minutes}
                      fmt="integer"
                      unit="min"
                    />
                  </dd>
                </div>
                <div className="activity-list__fact">
                  <dt>Steps</dt>
                  <dd>
                    <NumberValue value={activity.steps.length} fmt="integer" />
                  </dd>
                </div>
                <div className="activity-list__fact">
                  <dt>Prerequisites</dt>
                  <dd>{activity.prerequisites}</dd>
                </div>
              </dl>

              <Disclosure
                title="What you will learn"
                icon="book"
                count={activity.learning_points.length}
              >
                <ul className="activity-list__points">
                  {activity.learning_points.map((point, index) => (
                    <li key={index}>{point}</li>
                  ))}
                </ul>
              </Disclosure>

              <div className="activity-list__foot">
                <Link
                  className="activity-list__start"
                  to={`${ROUTES.activities}/${activity.id}`}
                >
                  Start: {activity.title}
                </Link>
                <Badge tone="warning" srPrefix="Review status: " icon="warning">
                  {activity.review_status}
                </Badge>
              </div>
            </Panel>
          </li>
        ))}
      </ul>
    </div>
  )
}
