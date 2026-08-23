/**
 * A generic table for an aggregate report row set.
 *
 * The administrator API returns each report table as a plain list of objects
 * whose shape is decided by the SQL that produced it, not by a schema the
 * frontend knows. Rather than hand-write a component per table — seven in the
 * usage tab alone — this derives its columns from the rows.
 *
 * Two decisions worth stating.
 *
 * 1. A column is numeric only when every value in it that is present is a
 *    number. One stray string demotes the whole column to text, which is the
 *    safe direction: right-aligning a column of words looks broken, whereas
 *    left-aligning a column of numbers merely looks plain.
 * 2. The unit and precision come from the column *name*. `mean_duration_ms` is
 *    milliseconds, `return_rate` is a fraction to be shown as a percentage,
 *    and a column of whole numbers is shown without a decimal part. The
 *    alternative — printing whatever the database returned — is how `2e-12`
 *    reaches a reader, and it does not happen here.
 *
 * The CSV written by the download button carries machine values from
 * `rawValue`, never the display strings, so a rounded figure on screen can
 * never end up in someone's spreadsheet.
 */

import { useId, useMemo } from 'react'
import type { ReactNode } from 'react'

import type { NumberFormatName } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Button, EmptyState, Icon } from '@/components/ui'
import { quoteField } from '@/components/results/csv'
import { rawValue } from '@/formatting/number'
import './ReportTable.css'

/** Rows rendered on screen. Everything beyond this is in the CSV. */
export const REPORT_ROW_CAP = 200

export type ReportRow = Record<string, unknown>

export interface ReportColumn {
  key: string
  label: string
  unit: string
  fmt: NumberFormatName
  precision: number | null
  numeric: boolean
}

/**
 * Column names whose plain de-underscoring reads badly, or which carry a scale
 * the reader needs (a 1–5 rating means nothing without its range).
 */
const LABEL_OVERRIDES: Record<string, string> = {
  clarity_rating: 'Clarity (1–5)',
  error_class: 'Error class',
  feature_category: 'Feature category',
  instrument_id: 'Instrument',
  max_score: 'Maximum score',
  mean_duration_ms: 'Mean duration',
  occurred_at: 'Occurred',
  org_type: 'Organisation type',
  organisation_code: 'Organisation code',
  participant_ref: 'Participant',
  post_confidence: 'Confidence after',
  post_date: 'Date',
  post_score: 'Score after',
  pre_confidence: 'Confidence before',
  pre_score: 'Score before',
  scenario_category: 'Scenario category',
  scenario_id: 'Scenario',
  submitted_date: 'Submitted',
  usability_rating: 'Usability (1–5)',
  usefulness_rating: 'Usefulness (1–5)',
  would_recommend: 'Would recommend (0–10)',
}

/**
 * Unit and presentation implied by a column-name suffix. Longest and most
 * specific first: `_ms` must be tested before `_m`.
 */
const UNIT_SUFFIXES: readonly {
  suffix: string
  unit: string
  fmt: NumberFormatName
  precision: number | null
}[] = [
  { suffix: '_ms', unit: 'ms', fmt: 'fixed', precision: 0 },
  { suffix: '_km', unit: 'km', fmt: 'significant', precision: 4 },
  { suffix: '_rate', unit: '', fmt: 'percent', precision: 1 },
  { suffix: '_pct', unit: '', fmt: 'percent', precision: 1 },
  { suffix: '_ratio', unit: '', fmt: 'significant', precision: 3 },
  { suffix: '_m', unit: 'm', fmt: 'distance', precision: null },
  { suffix: '_s', unit: 's', fmt: 'duration', precision: null },
]

function humanise(key: string): string {
  const override = LABEL_OVERRIDES[key]
  if (override !== undefined) return override
  const spaced = key.replace(/_/g, ' ').trim()
  if (spaced.length === 0) return key
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/** Every key present anywhere in the row set, in first-seen order. */
function collectKeys(rows: readonly ReportRow[]): string[] {
  const keys: string[] = []
  const seen = new Set<string>()
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key)
        keys.push(key)
      }
    }
  }
  return keys
}

function describeColumn(key: string, rows: readonly ReportRow[]): ReportColumn {
  let numbers = 0
  let others = 0
  let allIntegers = true

  for (const row of rows) {
    const value = row[key]
    if (value === null || value === undefined) continue
    if (typeof value === 'number' && Number.isFinite(value)) {
      numbers += 1
      if (!Number.isInteger(value)) allIntegers = false
    } else {
      others += 1
    }
  }

  const numeric = numbers > 0 && others === 0
  const label = humanise(key)

  if (!numeric) {
    return { key, label, unit: '', fmt: 'auto', precision: null, numeric: false }
  }

  for (const rule of UNIT_SUFFIXES) {
    if (key.endsWith(rule.suffix)) {
      return {
        key,
        label,
        unit: rule.unit,
        fmt: rule.fmt,
        precision: rule.precision,
        numeric: true,
      }
    }
  }

  // Counts are whole numbers and are shown as such; a mean is not.
  return allIntegers
    ? { key, label, unit: '', fmt: 'integer', precision: 0, numeric: true }
    : { key, label, unit: '', fmt: 'significant', precision: 4, numeric: true }
}

/** Derive the column set for a list of report rows. */
export function deriveColumns(
  rows: readonly ReportRow[],
  only?: readonly string[],
): ReportColumn[] {
  const available = collectKeys(rows)
  const keys =
    only === undefined ? available : only.filter((key) => available.includes(key))
  return keys.map((key) => describeColumn(key, rows))
}

/** One CSV cell. Numbers keep full machine precision and plain ASCII. */
function csvCell(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') return rawValue(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'string') return value
  return JSON.stringify(value) ?? ''
}

/**
 * Render report rows as an RFC 4180 document.
 *
 * Exported because the feedback tab downloads its full row set without ever
 * putting it on screen as a table.
 */
export function reportRowsToCsv(
  rows: readonly ReportRow[],
  columns: readonly ReportColumn[],
  comments: readonly string[] = [],
): string {
  const CRLF = '\r\n'
  const header = [
    '# Orbital Rendezvous Simulator - pilot administration export',
    ...comments.map((line) => `# ${line.replace(/[\r\n]+/g, ' ')}`),
    '# Anonymous and aggregate. Values are full machine precision, not the',
    '# rounded figures shown on screen.',
  ]
  const columnHeader = columns
    .map((column) =>
      quoteField(column.unit ? `${column.label} (${column.unit})` : column.label),
    )
    .join(',')
  const body = rows.map((row) =>
    columns.map((column) => quoteField(csvCell(row[column.key]))).join(','),
  )
  return [...header, columnHeader, ...body].join(CRLF) + CRLF
}

/**
 * Hand a generated CSV to the browser.
 *
 * Exported alongside `reportRowsToCsv` so a caller that builds its own row set
 * does not have to repeat the object-URL dance.
 */
export function downloadCsv(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function Cell({ value, column }: { value: unknown; column: ReportColumn }) {
  if (value === null || value === undefined) return <>—</>

  if (typeof value === 'number') {
    return (
      <NumberValue
        value={value}
        fmt={column.fmt}
        precision={column.precision}
        unit={column.unit}
        align={column.numeric ? 'right' : 'left'}
      />
    )
  }

  if (typeof value === 'boolean') return <>{value ? 'yes' : 'no'}</>
  if (typeof value === 'string') return <>{value === '' ? '—' : value}</>
  return <>{JSON.stringify(value)}</>
}

export interface ReportTableProps {
  /** Visible heading. Also names the CSV download button. */
  title: string
  /** The row set, exactly as the API returned it. */
  rows: readonly ReportRow[]
  /** Shown in place of the table when there are no rows. */
  emptyMessage: string
  /** Stem of the downloaded file, e.g. `usage_by_date` gives usage_by_date.csv. */
  downloadName: string
  /** Restrict and order the columns. Unknown keys are ignored. */
  columns?: readonly string[]
  /** Explanatory line under the heading. */
  caption?: ReactNode
  /** Explanatory line under the table. */
  footnote?: ReactNode
  /** Comment lines written above the data in the CSV, e.g. the active filters. */
  csvComments?: readonly string[]
  /** Rows rendered before the display is capped. */
  maxRows?: number
  headingLevel?: 2 | 3 | 4
}

export function ReportTable({
  title,
  rows,
  emptyMessage,
  downloadName,
  columns,
  caption,
  footnote,
  csvComments = [],
  maxRows = REPORT_ROW_CAP,
  headingLevel = 3,
}: ReportTableProps) {
  const headingId = useId()
  const Heading = `h${headingLevel}` as const

  const derived = useMemo(() => deriveColumns(rows, columns), [rows, columns])

  const total = rows.length
  const shown = Math.min(total, maxRows)
  const hidden = total - shown
  const visible = hidden > 0 ? rows.slice(0, maxRows) : rows

  function download() {
    downloadCsv(
      `${downloadName}.csv`,
      reportRowsToCsv(rows, derived, [`table: ${title}`, ...csvComments]),
    )
  }

  return (
    <section className="report-table" aria-labelledby={headingId}>
      <div className="report-table__header">
        <Heading className="report-table__heading" id={headingId}>
          <Icon name="table" size={15} />
          <span>{title}</span>
          {total > 0 ? (
            <span className="report-table__count">
              <NumberValue value={total} fmt="integer" precision={0} unit="" />
              <span>{total === 1 ? ' row' : ' rows'}</span>
            </span>
          ) : null}
        </Heading>

        {total > 0 ? (
          <Button size="sm" icon="download" onClick={download}>
            {`Download ${title.toLowerCase()} (CSV)`}
          </Button>
        ) : null}
      </div>

      {caption !== undefined ? (
        <p className="report-table__caption">{caption}</p>
      ) : null}

      {total === 0 || derived.length === 0 ? (
        <EmptyState icon="table" heading="Nothing to report" description={emptyMessage} compact />
      ) : (
        <>
          <div
            className="report-table__scroll"
            role="region"
            aria-labelledby={headingId}
            tabIndex={0}
          >
            <table className="report-table__table">
              <caption className="visually-hidden">{title}</caption>
              <thead>
                <tr>
                  {derived.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={
                        column.numeric
                          ? 'report-table__th report-table__th--numeric'
                          : 'report-table__th'
                      }
                    >
                      <span className="report-table__th-label">{column.label}</span>
                      {column.unit ? (
                        <span className="report-table__th-unit num--unit">
                          {column.unit}
                        </span>
                      ) : null}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visible.map((row, index) => (
                  <tr key={index}>
                    {derived.map((column) => (
                      <td
                        key={column.key}
                        className={
                          column.numeric
                            ? 'report-table__td report-table__td--numeric num'
                            : 'report-table__td'
                        }
                      >
                        <Cell value={row[column.key]} column={column} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {hidden > 0 ? (
            <p className="report-table__note">
              Showing the first{' '}
              <NumberValue value={shown} fmt="integer" precision={0} unit="" /> rows.{' '}
              <NumberValue value={hidden} fmt="integer" precision={0} unit="" /> further{' '}
              {hidden === 1 ? 'row is' : 'rows are'} not displayed. The CSV download
              contains all{' '}
              <NumberValue value={total} fmt="integer" precision={0} unit="" /> rows.
            </p>
          ) : null}
        </>
      )}

      {footnote !== undefined ? (
        <p className="report-table__note">{footnote}</p>
      ) : null}
    </section>
  )
}
