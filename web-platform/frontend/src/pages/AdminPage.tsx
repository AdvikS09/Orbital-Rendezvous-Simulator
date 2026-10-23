import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'

import { ApiError, api, downloadResponse } from '@/api/client'
import type { AdminReport, UserType } from '@/api/types'
import { MetricTiles } from '@/components/admin/MetricTiles'
import type { MetricTile } from '@/components/admin/MetricTiles'
import {
  ReportTable,
  deriveColumns,
  downloadCsv,
  reportRowsToCsv,
} from '@/components/admin/ReportTable'
import type { ReportRow } from '@/components/admin/ReportTable'
import { NumberValue } from '@/components/NumberValue'
import { rawValue } from '@/formatting/number'
import {
  Badge,
  Button,
  Callout,
  Disclosure,
  Icon,
  Panel,
  Spinner,
  TabPanel,
  Tabs,
} from '@/components/ui'
import type { TabItem } from '@/components/ui'
import './AdminPage.css'

const PAGE_TITLE = 'Pilot administration'

const PAGE_CAPTION =
  'Anonymous aggregate reporting. No participant can be identified from ' +
  'anything on this page.'

const SIGN_IN_TITLE = 'Administrator sign-in'

const SIGN_IN_NOTE =
  'Credentials are held in this tab only, for as long as it stays open. They ' +
  'are sent to the server on each request and are never written to browser ' +
  'storage.'

const NO_CREDENTIALS_CONFIGURED =
  'No administrator credentials are configured on this deployment. Set ' +
  'PILOT_ADMIN_USERNAME and PILOT_ADMIN_PASSWORD_HASH in the environment and ' +
  'restart. Generate a hash with python scripts/hash_password.py.'

const INCORRECT_CREDENTIALS =
  'Incorrect username or password. The sign-in locks for a period after ' +
  'repeated failed attempts.'

const DEMO_INCLUDED_WARNING =
  'Demo records are included. These figures are not pilot results.'

const INCLUDE_DEMO_HELP =
  'Off by default so demonstration and test runs are never counted as real ' +
  'pilot activity.'

const UNREGISTERED_CODE_NOTE =
  'Codes that were never registered by a coordinator all appear under ' +
  '"Unregistered pilot code". Use the cohort table below to tell them apart.'

const LEARNING_CAVEAT =
  'Only participants who completed both assessments appear here. The same ' +
  'instrument is used before and after, so part of any gain comes from having ' +
  'seen the questions already — treat these as indicative, not as a controlled ' +
  'measurement.'

const EXPORT_CAPTION =
  'One ZIP with every table above as CSV, plus a text file recording the ' +
  'metric definitions and the filters that produced them. Anonymous and ' +
  'aggregate throughout.'

const EXPORT_DEFINITIONS_NOTE =
  'The archive includes METRIC_DEFINITIONS.txt, so a spreadsheet separated ' +
  'from this page cannot drift from the meaning of its columns.'

const SECRETS_NOTE =
  'Secrets are never displayed. Database passwords are stripped from every ' +
  'label shown here and from every log line.'

const USER_TYPE_LABELS: Record<UserType, string> = {
  student: 'Student',
  teacher: 'Teacher',
  engineer_researcher: 'Engineer / researcher',
  other: 'Other',
}

const USER_TYPES: readonly UserType[] = [
  'student',
  'teacher',
  'engineer_researcher',
  'other',
]

type PeriodPreset = 'last_7' | 'last_30' | 'last_90' | 'all' | 'custom'

const PERIOD_OPTIONS: readonly { value: PeriodPreset; label: string }[] = [
  { value: 'last_7', label: 'Last 7 days' },
  { value: 'last_30', label: 'Last 30 days' },
  { value: 'last_90', label: 'Last 90 days' },
  { value: 'all', label: 'All time' },
  { value: 'custom', label: 'Custom range' },
]

const PRESET_DAYS: Record<'last_7' | 'last_30' | 'last_90', number> = {
  last_7: 7,
  last_30: 30,
  last_90: 90,
}

function isoDate(value: Date): string {
  const year = value.getFullYear()
  const month = `${value.getMonth() + 1}`.padStart(2, '0')
  const day = `${value.getDate()}`.padStart(2, '0')
  return `${year}-${month}-${day}`
}

function today(): string {
  return isoDate(new Date())
}

function daysAgo(days: number): string {
  const when = new Date()
  when.setDate(when.getDate() - days)
  return isoDate(when)
}

interface Credentials {
  username: string
  password: string
}

type ReportQuery = Record<string, string | number | boolean | undefined>

function numberFrom(source: Record<string, unknown> | undefined, key: string): number | null {
  const value = source?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function textFrom(row: ReportRow, key: string): string | null {
  const value = row[key]
  if (typeof value === 'string' && value.trim() !== '') return value
  return null
}

function meanOf(rows: readonly ReportRow[], key: string): number | null {
  let total = 0
  let count = 0
  for (const row of rows) {
    const value = row[key]
    if (typeof value === 'number' && Number.isFinite(value)) {
      total += value
      count += 1
    }
  }
  return count === 0 ? null : total / count
}

function countWhere(
  rows: readonly ReportRow[],
  key: string,
  predicate: (value: number) => boolean,
): number {
  let count = 0
  for (const row of rows) {
    const value = row[key]
    if (typeof value === 'number' && Number.isFinite(value) && predicate(value)) {
      count += 1
    }
  }
  return count
}

function authMessage(cause: unknown): string {
  if (cause instanceof ApiError) {
    if (cause.status === 503) return NO_CREDENTIALS_CONFIGURED
    if (cause.status === 401) return cause.body?.message ?? INCORRECT_CREDENTIALS
    return cause.message
  }
  return 'The request could not be completed.'
}

interface AdminSignInProps {
  query: ReportQuery
  onSignedIn: (credentials: Credentials, report: AdminReport) => void
  notice?: string | null
}

function AdminSignIn({ query, onSignedIn, notice }: AdminSignInProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setSubmitting(true)
    const credentials: Credentials = { username, password }
    try {
      const report = await api.adminReport(credentials, query)
      setPassword('')
      onSignedIn(credentials, report)
    } catch (cause) {
      setError(authMessage(cause))
      setSubmitting(false)
    }
  }

  return (
    <div className="admin-page__sign-in">
      <Panel title={SIGN_IN_TITLE} icon="lock" subtitle={PAGE_CAPTION}>
        {notice ? (
          <Callout severity="warning" className="admin-page__sign-in-notice">
            {notice}
          </Callout>
        ) : null}

        <form className="admin-page__form" onSubmit={submit} noValidate>
          <div className="admin-page__field">
            <label className="admin-page__label" htmlFor="admin-username">
              Username
            </label>
            <input
              id="admin-username"
              name="username"
              className="admin-page__input"
              type="text"
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              value={username}
              disabled={submitting}
              aria-invalid={error !== null || undefined}
              aria-describedby={error ? 'admin-sign-in-error' : undefined}
              onChange={(event) => setUsername(event.target.value)}
            />
          </div>

          <div className="admin-page__field">
            <label className="admin-page__label" htmlFor="admin-password">
              Password
            </label>
            <input
              id="admin-password"
              name="password"
              className="admin-page__input"
              type="password"
              autoComplete="current-password"
              value={password}
              disabled={submitting}
              aria-invalid={error !== null || undefined}
              aria-describedby={error ? 'admin-sign-in-error' : undefined}
              onChange={(event) => setPassword(event.target.value)}
            />
          </div>

          {error ? (
            <Callout severity="error" id="admin-sign-in-error">
              {error}
            </Callout>
          ) : null}

          <div className="admin-page__form-actions">
            <Button type="submit" variant="primary" icon="lock" loading={submitting}>
              Sign in
            </Button>
          </div>
        </form>

        <p className="admin-page__hint">{SIGN_IN_NOTE}</p>
      </Panel>
    </div>
  )
}

interface FilterSelectProps {
  id: string
  label: string
  value: string
  options: readonly { value: string; label: string }[]
  onChange: (value: string) => void
}

function FilterSelect({ id, label, value, options, onChange }: FilterSelectProps) {
  return (
    <div className="admin-page__field">
      <label className="admin-page__label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className="admin-page__select"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  )
}

const TAB_ITEMS_BASE: readonly Omit<TabItem, 'count'>[] = [
  { value: 'usage', label: 'Usage', icon: 'chart' },
  { value: 'learning', label: 'Learning impact', icon: 'book' },
  { value: 'feedback', label: 'Feedback', icon: 'user' },
  { value: 'export', label: 'Export', icon: 'download' },
  { value: 'system', label: 'System', icon: 'settings' },
]

const QUALITATIVE_FIELDS: readonly { key: string; label: string }[] = [
  { key: 'most_useful', label: 'Most useful' },
  { key: 'improvement', label: 'Would change' },
  { key: 'bug_report', label: 'Bug report' },
  { key: 'feature_request', label: 'Feature request' },
]

export function AdminPage() {
  const [auth, setAuth] = useState<Credentials | null>(null)
  const [signedOutReason, setSignedOutReason] = useState<string | null>(null)

  const [report, setReport] = useState<AdminReport | null>(null)
  const [organisations, setOrganisations] = useState<ReportRow[]>([])
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)

  const [tab, setTab] = useState('usage')

  const [preset, setPreset] = useState<PeriodPreset>('last_30')
  const [customStart, setCustomStart] = useState(() => daysAgo(30))
  const [customEnd, setCustomEnd] = useState(() => today())
  const [organisationId, setOrganisationId] = useState('')
  const [userType, setUserType] = useState('')
  const [includeDemo, setIncludeDemo] = useState(false)

  const { start, end } = useMemo(() => {
    if (preset === 'all') return { start: '', end: '' }
    if (preset === 'custom') return { start: customStart, end: customEnd }
    return { start: daysAgo(PRESET_DAYS[preset]), end: today() }
  }, [preset, customStart, customEnd])

  const query = useMemo<ReportQuery>(
    () => ({
      start: start || undefined,
      end: end || undefined,
      organisation_id: organisationId || undefined,
      user_type: userType || undefined,
      include_demo: includeDemo,
    }),
    [start, end, organisationId, userType, includeDemo],
  )

  const queryKey = JSON.stringify(query)

  const loadedKey = useRef<string | null>(null)

  const signOut = useCallback((reason: string | null = null) => {
    setAuth(null)
    setReport(null)
    setOrganisations([])
    setLoadError(null)
    setExportError(null)
    loadedKey.current = null
    setSignedOutReason(reason)
  }, [])

  const handleSignedIn = useCallback(
    (credentials: Credentials, first: AdminReport) => {
      loadedKey.current = queryKey
      setReport(first)
      setSignedOutReason(null)
      setAuth(credentials)
    },
    [queryKey],
  )

  useEffect(() => {
    if (auth === null) return undefined
    if (loadedKey.current === queryKey) return undefined

    let cancelled = false
    setLoading(true)
    setLoadError(null)

    api
      .adminReport(auth, query)
      .then((data) => {
        if (cancelled) return
        loadedKey.current = queryKey
        setReport(data)
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        if (cause instanceof ApiError && (cause.status === 401 || cause.status === 503)) {
          signOut(authMessage(cause))
          return
        }
        setLoadError(authMessage(cause))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [auth, query, queryKey, signOut])

  useEffect(() => {
    if (auth === null) return undefined
    let cancelled = false
    api
      .adminOrganisations(auth, includeDemo)
      .then((rows) => {
        if (!cancelled) setOrganisations(rows)
      })
      .catch(() => {
        if (!cancelled) setOrganisations([])
      })
    return () => {
      cancelled = true
    }
  }, [auth, includeDemo])

  const organisationOptions = useMemo(() => {
    const options: { value: string; label: string }[] = [
      { value: '', label: 'All organisations' },
    ]
    for (const row of organisations) {
      const id = row['id']
      const name = row['name']
      const code = row['code']
      if (typeof id !== 'number' || typeof name !== 'string') continue
      options.push({
        value: rawValue(id),
        label: typeof code === 'string' ? `${name} (${code})` : name,
      })
    }
    return options
  }, [organisations])

  const userTypeOptions = useMemo(
    () => [
      { value: '', label: 'All user types' },
      ...USER_TYPES.map((value) => ({ value, label: USER_TYPE_LABELS[value] })),
    ],
    [],
  )

  const organisationLabel =
    organisationOptions.find((option) => option.value === organisationId)?.label ??
    'All organisations'

  const userTypeLabel =
    userTypeOptions.find((option) => option.value === userType)?.label ?? 'All user types'

  const csvComments = useMemo(
    () => [
      `period: ${start || 'all time'} to ${end || 'today'}`,
      `organisation: ${organisationLabel}`,
      `user type: ${userTypeLabel}`,
      `demo records: ${includeDemo ? 'INCLUDED - NOT PILOT DATA' : 'excluded'}`,
    ],
    [start, end, organisationLabel, userTypeLabel, includeDemo],
  )

  async function exportEverything() {
    if (auth === null) return
    setExporting(true)
    setExportError(null)
    try {
      const response = await api.adminExport(auth, query)
      await downloadResponse(response, `pilot_export_${today()}.zip`)
    } catch (cause) {
      setExportError(authMessage(cause))
    } finally {
      setExporting(false)
    }
  }

  function downloadFeedback() {
    if (report === null) return
    const rows = report.feedback
    downloadCsv(
      'feedback.csv',
      reportRowsToCsv(rows, deriveColumns(rows), ['table: All feedback', ...csvComments]),
    )
  }

  if (auth === null) {
    return (
      <AdminSignIn query={query} onSignedIn={handleSignedIn} notice={signedOutReason} />
    )
  }

  const summary = report?.summary ?? null

  const summaryTiles: MetricTile[] = summary
    ? [
        {
          key: 'organisations',
          label: 'Organisations',
          value: summary.organisations,
          fmt: 'integer',
          precision: 0,
          help: 'Organisations with any recorded activity in this period.',
        },
        {
          key: 'unique_users',
          label: 'Unique users',
          value: summary.unique_users,
          fmt: 'integer',
          precision: 0,
          help: 'Anonymous participants who completed at least one simulation.',
        },
        {
          key: 'active_users',
          label: 'Active users',
          value: summary.active_users,
          fmt: 'integer',
          precision: 0,
          help: 'Participants who completed a simulation during the selected period.',
        },
        {
          key: 'returning_users',
          label: 'Returning users',
          value: summary.returning_users,
          fmt: 'integer',
          precision: 0,
          help: 'Participants who used the platform on at least two separate dates.',
        },
        {
          key: 'completed_runs',
          label: 'Completed runs',
          value: summary.completed_runs,
          fmt: 'integer',
          precision: 0,
          help: 'Simulations that finished successfully and displayed results.',
        },
        {
          key: 'failed_runs',
          label: 'Failed runs',
          value: summary.failed_runs,
          fmt: 'integer',
          precision: 0,
          help:
            'Simulations that raised an error during execution. Runs rejected by ' +
            'input validation or rate limiting are not failures — the engine never ran.',
        },
        {
          key: 'failure_rate',
          label: 'Failure rate',
          value: summary.failure_rate,
          fmt: 'percent',
          precision: 1,
          help: 'Failed runs as a fraction of every run that reached the engine.',
        },
        {
          key: 'runs_per_user',
          label: 'Simulations per user',
          value: summary.runs_per_user,
          fmt: 'fixed',
          precision: 1,
          help: 'Completed runs divided by unique users.',
        },
        {
          key: 'return_rate',
          label: 'Return rate',
          value: summary.return_rate,
          fmt: 'percent',
          precision: 1,
          help: 'Returning users as a fraction of unique users.',
        },
        {
          key: 'measured_impact_users',
          label: 'Measured-impact users',
          value: summary.measured_impact_users,
          fmt: 'integer',
          precision: 0,
          help: 'Participants who completed both the pre- and the post-assessment.',
        },
        {
          key: 'feedback_submissions',
          label: 'Feedback submissions',
          value: summary.feedback_submissions,
          fmt: 'integer',
          precision: 0,
          help: 'Feedback forms submitted during the selected period.',
        },
      ]
    : []

  const assessmentRows: readonly ReportRow[] = report?.assessment_changes ?? []
  const feedbackRows: readonly ReportRow[] = report?.feedback ?? []
  const ratings = report?.feedback_ratings

  const tabItems: TabItem[] = TAB_ITEMS_BASE.map((item) => {
    if (item.value === 'learning' && assessmentRows.length > 0) {
      return { ...item, count: assessmentRows.length }
    }
    if (item.value === 'feedback' && feedbackRows.length > 0) {
      return { ...item, count: feedbackRows.length }
    }
    return { ...item }
  })

  return (
    <div className="admin-page">
      <header className="admin-page__header">
        <div className="admin-page__title-group">
          <h1 className="admin-page__title">{PAGE_TITLE}</h1>
          <p className="admin-page__caption">{PAGE_CAPTION}</p>
        </div>
        <div className="admin-page__header-actions">
          {report ? (
            <Badge tone={report.is_demo_environment ? 'warning' : 'neutral'} icon="shield">
              {report.environment}
            </Badge>
          ) : null}
          <Button icon="close" onClick={() => signOut(null)}>
            Sign out
          </Button>
        </div>
      </header>

      <div className="admin-page__body">
        <aside className="admin-page__sidebar" aria-labelledby="admin-filters-heading">
          <h2 className="admin-page__sidebar-heading" id="admin-filters-heading">
            Report filters
          </h2>

          <FilterSelect
            id="admin-period"
            label="Period"
            value={preset}
            options={PERIOD_OPTIONS}
            onChange={(value) => setPreset(value as PeriodPreset)}
          />

          {preset === 'custom' ? (
            <fieldset className="admin-page__fieldset">
              <legend className="admin-page__legend">Date range</legend>
              <div className="admin-page__field">
                <label className="admin-page__label" htmlFor="admin-start">
                  From
                </label>
                <input
                  id="admin-start"
                  className="admin-page__input"
                  type="date"
                  value={customStart}
                  max={customEnd}
                  onChange={(event) => setCustomStart(event.target.value)}
                />
              </div>
              <div className="admin-page__field">
                <label className="admin-page__label" htmlFor="admin-end">
                  To
                </label>
                <input
                  id="admin-end"
                  className="admin-page__input"
                  type="date"
                  value={customEnd}
                  min={customStart}
                  onChange={(event) => setCustomEnd(event.target.value)}
                />
              </div>
            </fieldset>
          ) : null}

          <FilterSelect
            id="admin-organisation"
            label="Organisation"
            value={organisationId}
            options={organisationOptions}
            onChange={setOrganisationId}
          />

          <FilterSelect
            id="admin-user-type"
            label="User type"
            value={userType}
            options={userTypeOptions}
            onChange={setUserType}
          />

          <div className="admin-page__check">
            <div className="admin-page__check-row">
              <input
                id="admin-include-demo"
                className="admin-page__checkbox"
                type="checkbox"
                checked={includeDemo}
                aria-describedby="admin-include-demo-help"
                onChange={(event) => setIncludeDemo(event.target.checked)}
              />
              <label className="admin-page__check-label" htmlFor="admin-include-demo">
                Include development / demo records
              </label>
            </div>
            <p className="admin-page__help" id="admin-include-demo-help">
              {INCLUDE_DEMO_HELP}
            </p>
          </div>

          {includeDemo ? (
            <Callout severity="warning">{DEMO_INCLUDED_WARNING}</Callout>
          ) : null}

          <p className="admin-page__help">
            Environment: <code>{report?.environment ?? 'unknown'}</code>
          </p>
        </aside>

        <div className="admin-page__content">
          {includeDemo ? (
            <Callout severity="warning" title="Demo records are included">
              These figures are not pilot results. Every table and export below
              counts development and demonstration activity alongside real use.
            </Callout>
          ) : null}

          {report && report.is_demo_environment ? (
            <Callout severity="info" title="Not a production deployment">
              This deployment is running as <code>{report.environment}</code>, so every
              record it creates is flagged as demo data and excluded from reports by
              default. Set <code>PILOT_ENV=production</code> for a real pilot.
            </Callout>
          ) : null}

          {loadError ? (
            <Callout severity="error" title="The report could not be loaded">
              {loadError}
            </Callout>
          ) : null}

          {loading && report === null ? (
            <div className="admin-page__loading">
              <Spinner size="md" label="Loading the report" />
            </div>
          ) : null}

          {report ? (
            <>
              <div className="admin-page__summary" aria-busy={loading || undefined}>
                <MetricTiles
                  tiles={summaryTiles}
                  ariaLabel="Headline pilot metrics"
                  columns={4}
                />
                {loading ? (
                  <p className="admin-page__help">
                    <Spinner label="Refreshing" /> Refreshing for the current filters.
                  </p>
                ) : null}
              </div>

              <Tabs
                items={tabItems}
                value={tab}
                onChange={setTab}
                label="Report sections"
                idPrefix="admin"
              />

              <TabPanel value="usage" selected={tab} idPrefix="admin">
                <div className="admin-page__stack">
                  <ReportTable
                    title="Usage by date"
                    rows={report.usage_by_date}
                    emptyMessage="No activity recorded in this period."
                    downloadName="usage_by_date"
                    csvComments={csvComments}
                  />
                  <ReportTable
                    title="Participating organisations"
                    rows={report.usage_by_organisation}
                    emptyMessage="No organisations have recorded activity in this period."
                    downloadName="usage_by_organisation"
                    csvComments={csvComments}
                    footnote={UNREGISTERED_CODE_NOTE}
                  />
                  <ReportTable
                    title="Cohorts (by pilot code)"
                    rows={report.usage_by_pilot_code}
                    emptyMessage="No pilot codes have recorded activity in this period."
                    downloadName="usage_by_pilot_code"
                    csvComments={csvComments}
                  />
                  <ReportTable
                    title="Usage by user type"
                    rows={report.usage_by_user_type}
                    emptyMessage="No usage recorded."
                    downloadName="usage_by_user_type"
                    csvComments={csvComments}
                  />
                  <ReportTable
                    title="Usage by scenario"
                    rows={report.usage_by_scenario}
                    emptyMessage="No scenarios have been run in this period."
                    downloadName="usage_by_scenario"
                    csvComments={csvComments}
                  />
                  <ReportTable
                    title="Usage by feature category"
                    rows={report.usage_by_feature}
                    emptyMessage="No feature usage recorded."
                    downloadName="usage_by_feature"
                    csvComments={csvComments}
                  />
                  <ReportTable
                    title="Simulations per user"
                    rows={report.simulations_per_user}
                    emptyMessage="No participant has completed a simulation in this period."
                    downloadName="simulations_per_user"
                    csvComments={csvComments}
                  />
                </div>
              </TabPanel>

              <TabPanel value="learning" selected={tab} idPrefix="admin">
                <LearningPanel rows={assessmentRows} csvComments={csvComments} />
              </TabPanel>

              <TabPanel value="feedback" selected={tab} idPrefix="admin">
                <div className="admin-page__stack">
                  <FeedbackRatings ratings={ratings} />

                  <section
                    className="admin-page__section"
                    aria-labelledby="admin-qualitative"
                  >
                    <div className="admin-page__section-header">
                      <h3 className="admin-page__section-heading" id="admin-qualitative">
                        Qualitative feedback
                      </h3>
                      {feedbackRows.length > 0 ? (
                        <Button size="sm" icon="download" onClick={downloadFeedback}>
                          Download all feedback (CSV)
                        </Button>
                      ) : null}
                    </div>
                    <QualitativeFeedback rows={feedbackRows} />
                  </section>

                  <ReportTable
                    title="Bugs and feature requests only"
                    rows={report.bugs_and_requests}
                    emptyMessage="No bugs or feature requests reported in this period."
                    downloadName="bugs_and_requests"
                    columns={[
                      'submitted_date',
                      'user_type',
                      'organisation',
                      'bug_report',
                      'feature_request',
                    ]}
                    csvComments={csvComments}
                  />
                </div>
              </TabPanel>

              <TabPanel value="export" selected={tab} idPrefix="admin">
                <Panel title="Export everything" headingLevel={3} icon="download">
                  <p className="admin-page__prose">{EXPORT_CAPTION}</p>
                  <p className="admin-page__prose">{EXPORT_DEFINITIONS_NOTE}</p>

                  {exportError ? (
                    <Callout severity="error" title="The export failed">
                      {exportError}
                    </Callout>
                  ) : null}

                  <div className="admin-page__form-actions">
                    <Button
                      variant="primary"
                      icon="download"
                      loading={exporting}
                      loadingLabel="Building the archive"
                      onClick={exportEverything}
                    >
                      Download every table (ZIP)
                    </Button>
                  </div>
                </Panel>
              </TabPanel>

              <TabPanel value="system" selected={tab} idPrefix="admin">
                <div className="admin-page__stack">
                  <Panel title="Deployment" headingLevel={3} icon="settings">
                    <dl className="admin-page__definitions">
                      <div className="admin-page__definition">
                        <dt>Environment</dt>
                        <dd>
                          <code>{report.environment}</code>
                        </dd>
                      </div>
                      <div className="admin-page__definition">
                        <dt>Records marked as demo data</dt>
                        <dd>{report.is_demo_environment ? 'yes' : 'no'}</dd>
                      </div>
                      <div className="admin-page__definition">
                        <dt>Demo records in this report</dt>
                        <dd>{report.include_demo ? 'included' : 'excluded'}</dd>
                      </div>
                    </dl>

                    {report.is_demo_environment ? (
                      <Callout severity="info" title="Demo data">
                        Every record this deployment creates is flagged as demo data and
                        excluded from reports by default. Set{' '}
                        <code>PILOT_ENV=production</code> for a real pilot.
                      </Callout>
                    ) : null}

                    <p className="admin-page__help">{SECRETS_NOTE}</p>
                  </Panel>

                  <Panel title="Configuration warnings" headingLevel={3} icon="warning">
                    {report.warnings.length === 0 ? (
                      <p className="admin-page__help">No configuration warnings.</p>
                    ) : (
                      <div className="admin-page__stack">
                        {report.warnings.map((warning) => (
                          <Callout severity="warning" key={warning}>
                            {warning}
                          </Callout>
                        ))}
                      </div>
                    )}
                  </Panel>

                  <Disclosure title="Metric definitions" icon="book">
                    <dl className="admin-page__definitions">
                      {Object.entries(report.metric_definitions).map(([key, text]) => (
                        <div className="admin-page__definition" key={key}>
                          <dt>
                            <code>{key}</code>
                          </dt>
                          <dd>{text}</dd>
                        </div>
                      ))}
                    </dl>
                  </Disclosure>
                </div>
              </TabPanel>
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function LearningPanel({
  rows,
  csvComments,
}: {
  rows: readonly ReportRow[]
  csvComments: readonly string[]
}) {
  const meanBefore = meanOf(rows, 'pre_score')
  const meanAfter = meanOf(rows, 'post_score')
  const meanChange = meanOf(rows, 'change')
  const improved = countWhere(rows, 'change', (value) => value > 0)
  const unchanged = countWhere(rows, 'change', (value) => value === 0)
  const lower = countWhere(rows, 'change', (value) => value < 0)

  const confidenceBefore = meanOf(rows, 'pre_confidence')
  const confidenceAfter = meanOf(rows, 'post_confidence')

  const tiles: MetricTile[] = [
    {
      key: 'measured',
      label: 'Participants measured',
      value: rows.length,
      fmt: 'integer',
      precision: 0,
      help: 'Participants who completed both the pre- and the post-assessment.',
    },
    {
      key: 'mean_before',
      label: 'Mean before',
      value: meanBefore,
      fmt: 'fixed',
      precision: 2,
      help: 'Mean pre-assessment score across the participants shown here.',
    },
    {
      key: 'mean_after',
      label: 'Mean after',
      value: meanAfter,
      fmt: 'fixed',
      precision: 2,
      help: 'Mean post-assessment score across the same participants.',
    },
    {
      key: 'mean_change',
      label: 'Mean change',
      value: meanChange,
      fmt: 'fixed',
      precision: 2,
      help: 'Mean of the per-participant difference, post minus pre.',
    },
  ]

  return (
    <div className="admin-page__stack">
      <section className="admin-page__section" aria-labelledby="admin-learning">
        <h3 className="admin-page__section-heading" id="admin-learning">
          Pre- and post-assessment change
        </h3>
        <p className="admin-page__prose">{LEARNING_CAVEAT}</p>
      </section>

      {rows.length === 0 ? (
        <Callout severity="info">
          No participant has completed both assessments in this period.
        </Callout>
      ) : (
        <>
          <MetricTiles tiles={tiles} ariaLabel="Assessment change" columns={4} />

          <p className="admin-page__help">
            Improved: <NumberValue value={improved} fmt="integer" precision={0} unit="" />{' '}
            · unchanged:{' '}
            <NumberValue value={unchanged} fmt="integer" precision={0} unit="" /> · lower:{' '}
            <NumberValue value={lower} fmt="integer" precision={0} unit="" />
          </p>

          {confidenceBefore !== null && confidenceAfter !== null ? (
            <p className="admin-page__help">
              Mean self-rated confidence:{' '}
              <NumberValue value={confidenceBefore} fmt="fixed" precision={2} unit="" />{' '}
              before,{' '}
              <NumberValue value={confidenceAfter} fmt="fixed" precision={2} unit="" />{' '}
              after (1–5).
            </p>
          ) : null}

          <ReportTable
            title="Assessment changes"
            rows={rows}
            emptyMessage="No participant has completed both assessments in this period."
            downloadName="assessment_changes"
            csvComments={csvComments}
          />
        </>
      )}
    </div>
  )
}

function FeedbackRatings({ ratings }: { ratings: Record<string, unknown> | undefined }) {
  const responses = numberFrom(ratings, 'responses') ?? 0

  if (responses === 0) {
    return (
      <section className="admin-page__section" aria-labelledby="admin-ratings">
        <h3 className="admin-page__section-heading" id="admin-ratings">
          Ratings
        </h3>
        <Callout severity="info">No feedback has been submitted in this period.</Callout>
      </section>
    )
  }

  const overall: MetricTile[] = [
    {
      key: 'responses',
      label: 'Responses',
      value: responses,
      fmt: 'integer',
      precision: 0,
      help: 'Feedback forms submitted in this period.',
    },
    {
      key: 'usefulness',
      label: 'Usefulness (1–5)',
      value: numberFrom(ratings, 'mean_usefulness'),
      fmt: 'fixed',
      precision: 2,
      help: 'Mean self-reported usefulness across every respondent.',
    },
    {
      key: 'usability',
      label: 'Usability (1–5)',
      value: numberFrom(ratings, 'mean_usability'),
      fmt: 'fixed',
      precision: 2,
      help: 'Mean self-reported ease of use across every respondent.',
    },
    {
      key: 'recommend',
      label: 'Would recommend (0–10)',
      value: numberFrom(ratings, 'mean_recommend'),
      fmt: 'fixed',
      precision: 2,
      help: 'Mean answer to the recommendation question.',
    },
  ]

  const professional: MetricTile[] = [
    {
      key: 'professional_responses',
      label: 'Responses',
      value: numberFrom(ratings, 'professional_responses') ?? 0,
      fmt: 'integer',
      precision: 0,
      help: 'Respondents who identified themselves as an engineer or researcher.',
    },
    {
      key: 'professional_usefulness',
      label: 'Professional usefulness',
      value: numberFrom(ratings, 'professional_mean_usefulness'),
      fmt: 'fixed',
      precision: 2,
      help: 'Mean usefulness among engineers and researchers only.',
    },
    {
      key: 'professional_usability',
      label: 'Professional usability',
      value: numberFrom(ratings, 'professional_mean_usability'),
      fmt: 'fixed',
      precision: 2,
      help: 'Mean usability among engineers and researchers only.',
    },
  ]

  return (
    <div className="admin-page__stack">
      <MetricTiles tiles={overall} title="Ratings" headingLevel={3} columns={4} />
      <MetricTiles
        tiles={professional}
        title="Engineers and researchers specifically"
        headingLevel={3}
        columns={3}
      />
    </div>
  )
}

function QualitativeFeedback({ rows }: { rows: readonly ReportRow[] }) {
  const withText = rows.filter((row) =>
    QUALITATIVE_FIELDS.some((field) => textFrom(row, field.key) !== null),
  )

  if (withText.length === 0) {
    return (
      <Callout severity="info">
        No written comments have been submitted in this period.
      </Callout>
    )
  }

  return (
    <ul className="admin-page__cards" role="list">
      {withText.map((row, index) => {
        const rawType = row['user_type']
        const typeLabel =
          typeof rawType === 'string' && rawType in USER_TYPE_LABELS
            ? USER_TYPE_LABELS[rawType as UserType]
            : '—'
        return (
          <li className="admin-page__card" key={index}>
            <p className="admin-page__card-meta">
              <Icon name="clock" size={12} />
              <span>{textFrom(row, 'submitted_date') ?? '—'}</span>
              <span aria-hidden="true">·</span>
              <span>{typeLabel}</span>
              <span aria-hidden="true">·</span>
              <span>{textFrom(row, 'organisation') ?? '—'}</span>
              <span aria-hidden="true">·</span>
              <span>participant {textFrom(row, 'participant_ref') ?? '—'}</span>
            </p>
            <dl className="admin-page__card-body">
              {QUALITATIVE_FIELDS.map((field) => {
                const text = textFrom(row, field.key)
                if (text === null) return null
                return (
                  <div className="admin-page__card-entry" key={field.key}>
                    <dt>{field.label}</dt>
                    <dd>{text}</dd>
                  </div>
                )
              })}
            </dl>
          </li>
        )
      })}
    </ul>
  )
}
