from __future__ import annotations

import pytest

from app.services.formatting import (
    MINUS,
    NBSP,
    PLACEHOLDER,
    TIMES,
    format_number,
    format_scientific,
    format_significant,
    to_superscript,
)

SHARED_CASES = [
    (2e-12, 2, f"2.0{NBSP}{TIMES}{NBSP}10⁻¹²"),
    (1.1060555202205516e-13, 3, f"1.11{NBSP}{TIMES}{NBSP}10⁻¹³"),
    (3.986004418e14, 4, f"3.986{NBSP}{TIMES}{NBSP}10¹⁴"),
    (3.14159, 3, "3.14"),
    (0.0, 3, "0.00"),
]


@pytest.mark.parametrize("value,significant,expected", SHARED_CASES)
def test_scientific_matches_the_shared_table(value, significant, expected):
    assert format_scientific(value, significant) == expected


def test_the_defect_this_migration_fixes():
    output = format_scientific(2e-12, 2)
    assert "e-" not in output
    assert "E-" not in output
    assert output == f"2.0{NBSP}{TIMES}{NBSP}10⁻¹²"


def test_superscripts():
    assert to_superscript(-12) == "⁻¹²"
    assert to_superscript(14) == "¹⁴"
    assert to_superscript(0) == "⁰"


def test_unicode_minus_not_hyphen():
    assert format_scientific(-4.2e-5, 2).startswith(MINUS)
    assert not format_scientific(-4.2e-5, 2).startswith("-")


def test_auto_notation_thresholds():
    assert TIMES in format_number(1.5e-7)
    assert TIMES in format_number(6378137.0)
    assert TIMES not in format_number(2.3257791301763304)
    assert format_number(2.3257791301763304) == "2.326"


def test_missing_values():
    assert format_number(None) == PLACEHOLDER
    assert format_number(float("nan")) == PLACEHOLDER
    assert format_number(float("inf")) == PLACEHOLDER


def test_units_use_a_non_breaking_space():
    assert format_number(2.326, unit="m/s") == f"2.326{NBSP}m/s"


def test_significant_figures():
    assert format_significant(2.3257791301763304, 4) == "2.326"


def test_no_output_contains_machine_notation():
    for value in (1e-30, 2e-12, 1.5e-7, 0.001, 1.0, 5553.62, 1e5, 6.6743e-11):
        rendered = format_number(value)
        assert "e-" not in rendered and "e+" not in rendered, (
            f"{value} rendered as {rendered!r}"
        )
