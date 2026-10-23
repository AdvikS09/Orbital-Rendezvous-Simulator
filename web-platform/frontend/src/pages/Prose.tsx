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

        return (
          <p className="prose__paragraph" key={blockIndex}>
            {renderInlineMarkdown(lines.join(' '))}
          </p>
        )
      })}
    </div>
  )
}
