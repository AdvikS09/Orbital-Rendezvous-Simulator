/**
 * Client-side CSV generation for result tables.
 *
 * Two rules govern this file.
 *
 * 1. Exported numbers are the machine values, never the display strings. A
 *    figure rendered as `2.0 × 10⁻¹²` is exported as `2e-12`, because someone
 *    will open the file in a spreadsheet and compute with it. `rawValue()` from
 *    the formatter is the only path a number takes into a cell.
 * 2. The file says where it came from. A CSV that has been emailed twice and
 *    renamed once is worthless without its scenario, units and solver, so a
 *    comment header carries them.
 *
 * Quoting follows RFC 4180: fields containing a comma, a double quote or a
 * line break are wrapped in double quotes and internal quotes are doubled.
 * Records are separated by CRLF, which is what the standard specifies and what
 * Excel expects.
 */

import type { ColumnSpec, TableSpec } from '@/api/types'
import { rawValue } from '@/formatting/number'

/** Provenance written into the comment header above the data. */
export interface CsvContext {
  scenarioId?: string | null
  scenarioTitle?: string | null
  units?: string | null
  solver?: string | null
  /** ISO 8601 instant the result was produced. Defaults to now. */
  timestamp?: string | null
}

const CRLF = '\r\n'
const NEEDS_QUOTING = /[",\r\n]/

/** RFC 4180 field quoting. Leading or trailing spaces are preserved by quoting. */
export function quoteField(field: string): string {
  if (NEEDS_QUOTING.test(field) || field !== field.trim()) {
    return `"${field.replace(/"/g, '""')}"`
  }
  return field
}

/**
 * One cell, as it should appear in the file.
 *
 * Numeric columns go through `rawValue`, which preserves full precision and
 * uses plain ASCII: no unicode minus, no superscripts, no thousands
 * separators. Anything a spreadsheet would misread is not written.
 */
function cellText(value: unknown, column: ColumnSpec): string {
  if (value === null || value === undefined) return ''

  if (column.numeric) {
    if (typeof value === 'number') return rawValue(value)
    // A numeric column carrying a string is unusual but not impossible; keep
    // the original text rather than coercing it to NaN and losing it.
    const parsed = Number(value)
    return Number.isFinite(parsed) ? rawValue(parsed) : String(value)
  }

  if (typeof value === 'number') return rawValue(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'string') return value
  return JSON.stringify(value) ?? ''
}

function commentLine(label: string, value: string | null | undefined): string | null {
  const text = (value ?? '').trim()
  if (!text) return null
  // Strip line breaks so a multi-line value cannot break out of the comment.
  return `# ${label}: ${text.replace(/[\r\n]+/g, ' ')}`
}

/**
 * Render a table as an RFC 4180 CSV document.
 *
 * The header block is a run of `#` comment lines. Most tools ignore them;
 * pandas reads the file with `comment='#'`, and a human opening it in a text
 * editor sees the provenance first.
 */
export function toCsv(table: TableSpec, context: CsvContext = {}): string {
  const header = [
    '# Orbital Rendezvous Simulator - exported table',
    commentLine('table', `${table.label} (${table.id})`),
    commentLine(
      'scenario',
      context.scenarioTitle && context.scenarioId
        ? `${context.scenarioTitle} (${context.scenarioId})`
        : (context.scenarioTitle ?? context.scenarioId ?? null),
    ),
    commentLine('generated', context.timestamp ?? new Date().toISOString()),
    commentLine('units', context.units),
    commentLine('solver', context.solver),
    commentLine('caption', table.caption),
    '# Values are full machine precision, not the rounded figures shown on screen.',
  ].filter((line): line is string => line !== null)

  const columnHeader = table.columns
    .map((column) => quoteField(column.unit ? `${column.label} (${column.unit})` : column.label))
    .join(',')

  const body = table.rows.map((row) =>
    table.columns.map((column) => quoteField(cellText(row[column.key], column))).join(','),
  )

  return [...header, columnHeader, ...body].join(CRLF) + CRLF
}

/** A filesystem-safe name for a table export. */
export function csvFilename(table: TableSpec, scenarioId?: string | null): string {
  const stem = [scenarioId, table.id].filter(Boolean).join('_')
  return `${stem.replace(/[^A-Za-z0-9._-]+/g, '-')}.csv`
}
