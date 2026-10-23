import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'

import { Layout } from '@/components/Layout'
import { Spinner } from '@/components/ui/Spinner'
import { useSession } from '@/state/session'

const LandingPage = lazy(() =>
  import('@/pages/LandingPage').then((m) => ({ default: m.LandingPage })),
)
const ConsentPage = lazy(() =>
  import('@/pages/ConsentPage').then((m) => ({ default: m.ConsentPage })),
)
const SignInPage = lazy(() =>
  import('@/pages/SignInPage').then((m) => ({ default: m.SignInPage })),
)
const TrackPage = lazy(() =>
  import('@/pages/TrackPage').then((m) => ({ default: m.TrackPage })),
)
const ScenarioListPage = lazy(() =>
  import('@/pages/ScenarioListPage').then((m) => ({ default: m.ScenarioListPage })),
)
const ScenarioPage = lazy(() =>
  import('@/pages/ScenarioPage').then((m) => ({ default: m.ScenarioPage })),
)
const ActivityListPage = lazy(() =>
  import('@/pages/ActivityListPage').then((m) => ({ default: m.ActivityListPage })),
)
const ActivityPage = lazy(() =>
  import('@/pages/ActivityPage').then((m) => ({ default: m.ActivityPage })),
)
const AssessmentPage = lazy(() =>
  import('@/pages/AssessmentPage').then((m) => ({ default: m.AssessmentPage })),
)
const FeedbackPage = lazy(() =>
  import('@/pages/FeedbackPage').then((m) => ({ default: m.FeedbackPage })),
)
const ConfirmationPage = lazy(() =>
  import('@/pages/ConfirmationPage').then((m) => ({ default: m.ConfirmationPage })),
)
const AdminPage = lazy(() =>
  import('@/pages/AdminPage').then((m) => ({ default: m.AdminPage })),
)

function PageFallback() {
  return (
    <div style={{ display: 'grid', placeItems: 'center', minHeight: '40vh' }}>
      <Spinner label="Loading" />
    </div>
  )
}

function RequireConsent({ children }: { children: React.ReactNode }) {
  const { hasConsented } = useSession()
  const location = useLocation()
  if (!hasConsented) {
    return <Navigate to="/consent" state={{ from: location.pathname }} replace />
  }
  return <>{children}</>
}

export function App() {
  return (
    <Layout>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/consent" element={<ConsentPage />} />
          <Route path="/sign-in" element={<SignInPage />} />

          <Route
            path="/track"
            element={
              <RequireConsent>
                <TrackPage />
              </RequireConsent>
            }
          />
          <Route path="/scenarios" element={<ScenarioListPage />} />
          <Route path="/scenarios/:scenarioId" element={<ScenarioPage />} />
          <Route path="/activities" element={<ActivityListPage />} />
          <Route path="/activities/:activityId" element={<ActivityPage />} />
          <Route
            path="/assessment/:phase"
            element={
              <RequireConsent>
                <AssessmentPage />
              </RequireConsent>
            }
          />
          <Route
            path="/feedback"
            element={
              <RequireConsent>
                <FeedbackPage />
              </RequireConsent>
            }
          />
          <Route
            path="/done"
            element={
              <RequireConsent>
                <ConfirmationPage />
              </RequireConsent>
            }
          />

          <Route path="/admin" element={<AdminPage />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
