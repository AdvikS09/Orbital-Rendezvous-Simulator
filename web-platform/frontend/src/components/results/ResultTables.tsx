/**
 * Tabular output.
 *
 * Nothing here is virtualised. A trajectory can be tens of thousands of rows
 * and no one reads row 8,431 on screen; they download the file and open it in
 * something built for the job. So the display is capped, the cap is stated
 * plainly, and the note points at the download that contains everything.
 *
 * Wide tables scroll inside their own container. The page itself never scrolls
 * sideways, because a horizontally scrolling page hides content from anyone
 * who does not think to look for it.
 */

import { useId } from 'react'
import type { ColumnSpec, TableSpec } from '@/api/types'
import { Icon } from '@/components/ui/Icon'
import { NumberValue } from '@/components/NumberValue'
import './ResultTables.css'

/** Rows shown on screen. Everything beyond this is in the download. */
export const DEFAULT_ROW_CAP = 200

export interface ResultTablesProps {
  tables: TableSpec[]
  /** Maximum rows rendered per table. */
  maxRows?: number
  headingLevel?: 2 | 3 | 4
}

/** Narrow an arbitrary cell to something that can be rendered safely. */
function cellValue(value: unknown): number | string | null {
  if (value === null || value === undefined) return null
  if (typeof value === 'number' || typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  return JSON.stringify(value) ?? ''
}

function Cell({ value, column }: { value: unknown; column: ColumnSpec }) {
  const resolved = cellValue(value)

  // Numeric columns, and any stray number in a text column, go through the
  // formatter. A raw JavaScript number must never reach the DOM.
  if (column.numeric || typeof resolved === 'number') {
    return (
      <NumberValue value={resolved} fmt={column.fmt} precision={column.precision} unit="" />
    )
  }

  return <>{resolved ?? '—'}</>
}

function ResultTable({
  table,
  maxRows,
  headingLevel,
}: {
  table: TableSpec
  maxRows: number
  headingLevel: 2 | 3 | 4
}) {
  const headingId = useId()
  const Heading = `h${headingLevel}` as const

  const total = table.rows.length
  const shown = Math.min(total, maxRows)
  const hidden = total - shown
  const visibleRows = hidden > 0 ? table.rows.slice(0, maxRows) : table.rows

  return (
    <section className="result-tables__table-section" aria-labelledby={headingId}>
      <Heading className="result-tables__table-heading" id={headingId}>
        <Icon name="table" size={15} />
        <span>{table.label}</span>
        <span className="result-tables__count">
          <NumberValue value={total} fmt="integer" precision={0} unit="" />
          <span>{total === 1 ? ' row' : ' rows'}</span>
        </span>
      </Heading>

      {table.caption ? <p className="result-tables__caption">{table.caption}</p> : null}

      <div
        className="result-tables__scroll"
        role="region"
        aria-labelledby={headingId}
        tabIndex={0}
      >
        <table className="result-tables__table">
          <caption className="visually-hidden">
            {table.caption ? `${table.label}. ${table.caption}` : table.label}
          </caption>
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  className={
                    column.numeric
                      ? 'result-tables__header result-tables__header--numeric'
                      : 'result-tables__header'
                  }
                >
                  <span className="result-tables__header-label">{column.label}</span>
                  {column.unit ? (
                    <span className="result-tables__header-unit num--unit">{column.unit}</span>
                  ) : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr key={index}>
                {table.columns.map((column) => (
                  <td
                    key={column.key}
                    className={
                      column.numeric
                        ? 'result-tables__cell result-tables__cell--numeric num'
                        : 'result-tables__cell'
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
        <p className="result-tables__note">
          Showing the first{' '}
          <NumberValue value={shown} fmt="integer" precision={0} unit="" /> rows.{' '}
          <NumberValue value={hidden} fmt="integer" precision={0} unit="" /> further{' '}
          {hidden === 1 ? 'row is' : 'rows are'} not displayed. The CSV download below
          contains all{' '}
          <NumberValue value={total} fmt="integer" precision={0} unit="" /> rows at full
          precision.
        </p>
      ) : null}
    </section>
  )
}

export function ResultTables({
  tables,
  maxRows = DEFAULT_ROW_CAP,
  headingLevel = 3,
}: ResultTablesProps) {
  if (tables.length === 0) return null

  return (
    <div className="result-tables">
      {tables.map((table) => (
        <ResultTable
          key={table.id}
          table={table}
          maxRows={maxRows}
          headingLevel={headingLevel}
        />
      ))}
    </div>
  )
}
