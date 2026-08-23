"""Server-side number formatting, used only inside interpretation sentences.

Almost all formatting happens in the frontend. This exists for the handful of
places where a number is embedded in a *sentence* the backend composes -- for
example "finishing 1.50 x 10^-7 m from the target". Splitting that sentence
apart so React could format the middle of it would make the text harder to
maintain than the duplication is worth.

It mirrors ``web-platform/frontend/src/formatting/number.ts``, and
``tests/test_formatting_parity.py`` asserts the two agree, so they cannot drift.
"""

from __future__ import annotations

import math

SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "-": "⁻", "+": "⁺",
}

TIMES = "×"      # multiplication sign
NBSP = " "       # non-breaking space
MINUS = "−"      # true minus sign
PLACEHOLDER = "—"  # em dash

#: Magnitudes outside this range switch to scientific notation.
AUTO_LOWER = 1e-3
AUTO_UPPER = 1e5


def to_superscript(exponent: int | str) -> str:
    return "".join(SUPERSCRIPTS.get(character, character) for character in str(exponent))


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def format_scientific(value: float | None, significant: int = 4,
                      unit: str = "") -> str:
    """Render in scientific notation with Unicode superscripts."""
    if _missing(value):
        return PLACEHOLDER
    number = float(value)  # type: ignore[arg-type]
    if number == 0.0:
        body = f"{0:.{max(0, significant - 1)}f}"
    else:
        mantissa, exponent = f"{number:.{max(0, significant - 1)}e}".split("e")
        mantissa = mantissa.replace("-", MINUS)
        exponent_value = int(exponent)
        body = (mantissa if exponent_value == 0
                else f"{mantissa}{NBSP}{TIMES}{NBSP}10{to_superscript(exponent_value)}")
    return f"{body}{NBSP}{unit}".strip() if unit else body


def format_significant(value: float | None, significant: int = 4,
                       unit: str = "") -> str:
    if _missing(value):
        return PLACEHOLDER
    number = float(value)  # type: ignore[arg-type]
    precise = f"{number:.{significant}g}"
    if "e" in precise or "E" in precise:
        return format_scientific(number, significant, unit)
    body = precise.replace("-", MINUS)
    return f"{body}{NBSP}{unit}".strip() if unit else body


def format_number(value: float | None, significant: int = 4, unit: str = "") -> str:
    """Automatic notation: scientific for very small or very large magnitudes."""
    if _missing(value):
        return PLACEHOLDER
    number = float(value)  # type: ignore[arg-type]
    magnitude = abs(number)
    if number != 0.0 and (magnitude < AUTO_LOWER or magnitude >= AUTO_UPPER):
        return format_scientific(number, significant, unit)
    return format_significant(number, significant, unit)


def display_metric(value: float | None, unit: str = "", significant: int = 4) -> str:
    """Format a metric for embedding in an interpretation sentence."""
    return format_number(value, significant, unit)


__all__ = [
    "to_superscript", "format_scientific", "format_significant", "format_number",
    "display_metric", "TIMES", "NBSP", "MINUS", "PLACEHOLDER",
]
