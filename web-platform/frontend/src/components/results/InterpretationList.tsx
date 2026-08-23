/**
 * What the numbers mean, in the engine's own words.
 *
 * Interpretations arrive as short passages of text carrying a severity. They
 * are the part of a result that stops a figure being read as more certain than
 * it is, so they sit directly under the metrics rather than at the bottom of
 * the page.
 *
 * The text contains a deliberately tiny subset of Markdown: `**bold**`,
 * `*italic*`, and blank-line paragraph breaks. Nothing else is interpreted.
 * The renderer below builds React elements, so every other character reaches
 * the DOM as a text node and is escaped by React. There is no
 * `dangerouslySetInnerHTML` here and there must never be: the text is written
 * by the engine today, but a renderer that trusts its input is a hole waiting
 * for the day some of that text comes from a user-supplied scenario name.
 */

import type { ReactNode } from 'react'
import type { Interpretation } from '@/api/types'
import { Callout } from '@/components/ui/Callout'
import './InterpretationList.css'

/**
 * Bold before italic, so `**text**` is never mistaken for an empty emphasis
 * followed by a stray asterisk. Italic spans do not cross a line break.
 */
const INLINE_PATTERN = /\*\*([\s\S]+?)\*\*|\*([^*\n]+?)\*/g

/** Inline emphasis only. Everything unmatched is returned as plain text. */
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
  /** Class applied to each paragraph. */
  className?: string
}

/**
 * Blank-line separated paragraphs with inline emphasis.
 *
 * A single newline inside a paragraph is treated as a space, which is how
 * Markdown itself treats it, and keeps hard-wrapped source text from arriving
 * on screen with the wrapping baked in.
 */
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
  /** Optional visible heading above the group. */
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
