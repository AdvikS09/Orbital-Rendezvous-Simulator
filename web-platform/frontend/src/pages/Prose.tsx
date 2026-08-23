/**
 * Explanatory text supplied by the API, rendered as blocks.
 *
 * Scenario explanations and activity briefs are authored in the Python content
 * modules as light Markdown: paragraphs, bullet lists, the occasional heading
 * and bold run. Dropping that into a `<p>` as a single string would show the
 * asterisks; pulling in a full Markdown parser would be a dependency and an
 * injection surface for a handful of constructs.
 *
 * Inline emphasis is not reimplemented here — it comes from
 * `renderInlineMarkdown`, which the result panels already use, so there is one
 * implementation of what `**` means rather than two that can disagree. This
 * module adds only the block structure that longer prose needs and short
 * interpretation strings do not.
 *
 * As there, everything is built as React elements and never as HTML, so any
 * character the block rules do not recognise reaches the DOM as an escaped text
 * node. There is no `dangerouslySetInnerHTML` here and there must never be.
 */

import { renderInlineMarkdown } from '@/components/results/InterpretationList'
import './Prose.css'

const BULLET = /^[-*]\s+/
const HEADING = /^(#{1,4})\s+(.*)$/

export interface ProseProps {
  text: string | null | undefined
  className?: string
}

export function Prose({ text, className }: ProseProps) {
  const source = (text ?? '').trim()
  if (source === '') return null

  const blocks = source.split(/\n[ \t]*\n/)
  const classes = ['prose', className ?? ''].filter(Boolean).join(' ')

  return (
    <div className={classes}>
      {blocks.map((block, blockIndex) => {
        const lines = block
          .split('\n')
          .map((line) => line.trim())
          .filter((line) => line !== '')
        if (lines.length === 0) return null

        const first = lines[0] ?? ''
        const heading = HEADING.exec(first)
        if (heading !== null && lines.length === 1) {
          return (
            <h4 className="prose__heading" key={blockIndex}>
              {renderInlineMarkdown(heading[2] ?? '')}
            </h4>
          )
        }

        if (lines.every((line) => BULLET.test(line))) {
          return (
            <ul className="prose__list" key={blockIndex}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  {renderInlineMarkdown(line.replace(BULLET, ''))}
                </li>
              ))}
            </ul>
          )
        }

        // A single newline inside a paragraph is a space, as in Markdown, so
        // hard-wrapped source text does not arrive with the wrapping baked in.
        return (
          <p className="prose__paragraph" key={blockIndex}>
            {renderInlineMarkdown(lines.join(' '))}
          </p>
        )
      })}
    </div>
  )
}
