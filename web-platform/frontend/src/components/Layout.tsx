/**
 * Application shell: persistent sidebar navigation, main content region and
 * the standing status footer.
 *
 * On desktop the sidebar is always visible. Below 900px it collapses behind a
 * toggle and renders as an overlay drawer, so panels stack in a sensible order
 * rather than being squeezed.
 */

import { useCallback, useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

import { NumberValue } from '@/components/NumberValue'
import { Icon, type IconName } from '@/components/ui/Icon'
import { useSession } from '@/state/session'

import './Layout.css'

interface NavItem {
  to: string
  label: string
  icon: IconName
  end?: boolean
}

const PRIMARY_NAV: NavItem[] = [
  { to: '/', label: 'Overview', icon: 'book', end: true },
  { to: '/activities', label: 'Guided activities', icon: 'book' },
  { to: '/scenarios', label: 'Scenarios', icon: 'orbit' },
]

const SECONDARY_NAV: NavItem[] = [
  { to: '/feedback', label: 'Feedback', icon: 'chart' },
  { to: '/admin', label: 'Administration', icon: 'lock' },
]

/**
 * The single honest description of what this software is.
 * Mirrors ``STATUS_STATEMENT`` in pilot/ui/common.py so the two cannot drift.
 */
export const STATUS_STATEMENT =
  'This is open-source research and educational software. It is not ' +
  'NASA-approved, NASA-validated, flight-certified, or operational mission ' +
  'software, and it must not be used for operational flight decisions.'

export const VALIDATION_STATEMENT =
  'The simulation engine is checked against closed-form solutions and ' +
  'conservation laws, and those checks are part of the automated test suite. ' +
  'It has not been benchmarked against an external tool such as NASA GMAT or ' +
  'Ansys/AGI STK. The platform provides the workflow to carry out such a ' +
  'comparison yourself, but no such comparison is shipped with it.'

export function Layout({ children }: { children: React.ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const session = useSession()
  const location = useLocation()

  // Close the mobile drawer whenever the route changes.
  useEffect(() => {
    setDrawerOpen(false)
  }, [location.pathname])

  const closeDrawer = useCallback(() => setDrawerOpen(false), [])

  return (
    <div className="layout">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="layout__topbar">
        <button
          type="button"
          className="layout__menu-button"
          aria-label={drawerOpen ? 'Close navigation' : 'Open navigation'}
          aria-expanded={drawerOpen}
          aria-controls="primary-navigation"
          onClick={() => setDrawerOpen((open) => !open)}
        >
          <Icon name={drawerOpen ? 'close' : 'menu'} size={18} />
        </button>
        <span className="layout__wordmark">
          <Icon name="orbit" size={18} aria-hidden />
          Orbital Rendezvous Simulator
        </span>
        {session.isSignedIn ? (
          <span className="layout__session" title="Anonymous participant reference">
            <Icon name="user" size={14} aria-hidden />
            <code>{session.participantRef}</code>
          </span>
        ) : null}
      </header>

      {drawerOpen ? (
        <button
          type="button"
          className="layout__scrim"
          aria-label="Close navigation"
          onClick={closeDrawer}
        />
      ) : null}

      <nav
        id="primary-navigation"
        className={`layout__sidebar${drawerOpen ? ' layout__sidebar--open' : ''}`}
        aria-label="Primary"
      >
        <div className="layout__brand">
          <Icon name="orbit" size={22} aria-hidden />
          <div>
            <span className="layout__brand-title">Orbital Rendezvous</span>
            <span className="layout__brand-subtitle">Simulator</span>
          </div>
        </div>

        <ul className="layout__nav" role="list">
          {PRIMARY_NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `layout__nav-link${isActive ? ' layout__nav-link--active' : ''}`
                }
              >
                <Icon name={item.icon} size={16} aria-hidden />
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="layout__nav-divider" role="presentation" />

        <ul className="layout__nav" role="list">
          {SECONDARY_NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `layout__nav-link${isActive ? ' layout__nav-link--active' : ''}`
                }
              >
                <Icon name={item.icon} size={16} aria-hidden />
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="layout__sidebar-footer">
          {session.isSignedIn ? (
            <p className="layout__session-detail">
              Signed in anonymously as <code>{session.participantRef}</code>
              <br />
              {session.organisationName}
              <br />
              <NumberValue value={session.completedRuns} fmt="integer" />{' '}
              simulation{session.completedRuns === 1 ? '' : 's'} completed this
              visit.
            </p>
          ) : (
            <p className="layout__session-detail">Not signed in.</p>
          )}
          <p className="layout__disclaimer">
            Not NASA-approved. Not for operational use. MIT licence.
          </p>
        </div>
      </nav>

      <main id="main-content" className="layout__main" tabIndex={-1}>
        <div className="layout__content">{children}</div>
        <footer className="layout__footer">
          <details>
            <summary>About this software, its limits, and your data</summary>
            <p>{STATUS_STATEMENT}</p>
            <p>{VALIDATION_STATEMENT}</p>
          </details>
          <p className="layout__footer-note">
            Research and educational software — not NASA-approved and not for
            operational use.
          </p>
        </footer>
      </main>
    </div>
  )
}
