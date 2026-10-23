import { useId, useState } from 'react'
import type { ParamValues, SimulationResult } from '@/api/types'
import { ApiError, api, downloadResponse } from '@/api/client'
import { Icon } from '@/components/ui/Icon'
import { csvFilename, toCsv } from './csv'
import './ExportBar.css'

export interface ExportBarProps {
  result: SimulationResult
  scenarioId: string
  params: ParamValues
  headingLevel?: 2 | 3 | 4
}

const BYTE_ORDER_MARK = '\ufeff'

function downloadText(content: string, filename: string, mime: string, bom = false): void {
  const blob = new Blob([bom ? BYTE_ORDER_MARK + content : content], { type: mime })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function fileStem(scenarioId: string, completedAt: string): string {
  const stamp = completedAt.slice(0, 19).replace(/[:T]/g, '-')
  return `${scenarioId}_${stamp}`.replace(/[^A-Za-z0-9._-]+/g, '-')
}

export function ExportBar({ result, scenarioId, params, headingLevel = 2 }: ExportBarProps) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('')
  const headingId = useId()

  const Heading = `h${headingLevel}` as const
  const stem = fileStem(scenarioId, result.completed_at)
  const tables = result.tables.filter((table) => table.downloadable)

  async function run(id: string, label: string, action: () => void | Promise<void>) {
    setBusy(id)
    setError(null)
    setStatus('')
    try {
      await action()
      setStatus(`${label} downloaded.`)
    } catch (cause) {
      const message =
        cause instanceof ApiError
          ? cause.message
          : 'The download could not be prepared. Try again, or reload the page if it persists.'
      setError(message)
    } finally {
      setBusy(null)
    }
  }

  const csvContext = {
    scenarioId: result.scenario_id,
    scenarioTitle: result.scenario_title,
    units: result.solver.units,
    solver: `${result.solver.integrator} (engine ${result.solver.engine_version})`,
    timestamp: result.completed_at,
  }

  return (
    <section className="export-bar" aria-labelledby={headingId}>
      <Heading className="export-bar__heading" id={headingId}>
        Downloads
      </Heading>

      <div className="export-bar__buttons">
        {tables.map((table) => (
          <button
            key={table.id}
            type="button"
            className="export-bar__button"
            disabled={busy !== null}
            aria-busy={busy === `csv-${table.id}`}
            onClick={() =>
              void run(`csv-${table.id}`, `${table.label} (CSV)`, () => {
                downloadText(
                  toCsv(table, csvContext),
                  csvFilename(table, stem),
                  'text/csv;charset=utf-8',
                  true,
                )
              })
            }
          >
            <Icon name="download" size={15} />
            <span>{table.label} (CSV)</span>
          </button>
        ))}

        <button
          type="button"
          className="export-bar__button"
          disabled={busy !== null}
          aria-busy={busy === 'config'}
          onClick={() =>
            void run('config', 'Scenario configuration (JSON)', async () => {
              const response = await api.exportConfig(scenarioId, params)
              await downloadResponse(response, `${stem}_config.json`)
            })
          }
        >
          <Icon name="download" size={15} />
          <span>Scenario configuration (JSON)</span>
        </button>

        <button
          type="button"
          className="export-bar__button"
          disabled={busy !== null}
          aria-busy={busy === 'notes'}
          onClick={() =>
            void run('notes', 'Notes and caveats (TXT)', async () => {
              const response = await api.exportNotes(scenarioId, params, result.warnings)
              await downloadResponse(response, `${stem}_notes.txt`)
            })
          }
        >
          <Icon name="download" size={15} />
          <span>Notes and caveats (TXT)</span>
        </button>

        <button
          type="button"
          className="export-bar__button"
          disabled={busy !== null}
          aria-busy={busy === 'result'}
          onClick={() =>
            void run('result', 'Full result (JSON)', () => {
              downloadText(
                JSON.stringify(result, null, 2),
                `${stem}_result.json`,
                'application/json;charset=utf-8',
              )
            })
          }
        >
          <Icon name="download" size={15} />
          <span>Full result (JSON)</span>
        </button>
      </div>

      {error ? (
        <p className="export-bar__error" role="alert">
          <Icon name="error" size={15} />
          <span>{error}</span>
        </p>
      ) : null}

      <p className="visually-hidden" role="status">
        {status}
      </p>

      <p className="export-bar__caption">
        Files contain the numerical result, the parameters that produced it and the
        solver metadata. No participant, session or contact information is written to
        any download. CSV values are full machine precision, not the rounded figures
        shown on screen.
      </p>
    </section>
  )
}
