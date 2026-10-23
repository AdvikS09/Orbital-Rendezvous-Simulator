import type { ReactNode } from 'react'
import type { Interpretation } from '@/api/types'
import { Callout } from '@/components/ui/Callout'
import './InterpretationList.css'

const INLINE_PATTERN = /\*\*([\s\S]+?)\*\*|\*([^*\n]+?)\*/g

export function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const pattern = new RegExp(INLINE_PATTERN.source, 'g')
  let cursor = 0
  let key = 0
  let match = pattern.exec(text)

  while (match !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index))

    const bold = match[1]
    const italic = match[2]
    if (bold !== undefined) {
      nodes.push(<strong key={`b${key}`}>{bold}</strong>)
    } else if (italic !== undefined) {
      nodes.push(<em key={`i${key}`}>{italic}</em>)
    } else {
      nodes.push(match[0])
    }

    cursor = match.index + match[0].length
    key += 1
    match = pattern.exec(text)
  }

  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

export interface MarkdownOptions {
  className?: string
}

export function renderMarkdown(
  text: string | null | undefined,
  options: MarkdownOptions = {},
): ReactNode {
  const className = options.className ?? 'interpretation-list__paragraph'
  const paragraphs = (text ?? '')
    .split(/\n[ \t]*\n/)
    .map((paragraph) => paragraph.replace(/[ \t]*\n[ \t]*/g, ' ').trim())
    .filter((paragraph) => paragraph.length > 0)

  if (paragraphs.length === 0) return null

  return paragraphs.map((paragraph, index) => (
    <p className={className} key={index}>
      {renderInlineMarkdown(paragraph)}
    </p>
  ))
}

export interface InterpretationListProps {
  interpretations: Interpretation[]
  title?: string
  headingLevel?: 2 | 3 | 4
}

export function InterpretationList({
  interpretations,
  title = 'What this result means',
  headingLevel = 2,
}: InterpretationListProps) {
  if (interpretations.length === 0) return null

  const Heading = `h${headingLevel}` as const

  return (
    <section className="interpretation-list" aria-label={title}>
      <Heading className="interpretation-list__heading">{title}</Heading>
      <div className="interpretation-list__items">
        {interpretations.map((item, index) => (
          <Callout
            key={`${item.severity}-${index}`}
            severity={item.severity}
            title={item.title ?? undefined}
          >
            {renderMarkdown(item.text)}
          </Callout>
        ))}
      </div>
    </section>
  )
}
