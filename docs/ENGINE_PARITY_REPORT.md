# Engine parity report

Does the new FastAPI endpoint return the same physics as the existing engine?

**Result: yes, bit-for-bit.** 338,556 values compared across all 11 scenarios.
Maximum absolute difference 0.000 × 10⁰. Maximum relative difference
0.000 × 10⁰. No discrepancy of any size, in any scenario, in any quantity.

* Date: 2026-08-23
* Branch: `custom-web-migration`
* Environment: Windows 11, Python 3.14.6, numpy 2.4.6, pandas 3.0.3,
  FastAPI 0.141.1, pydantic 2.13.4
* Reproduce with `python tests/_parity_report.py` from `web-platform/backend`
* Enforced continuously by `web-platform/backend/tests/test_engine_parity.py`
  (116 tests)

---

## 1. Method

For each scenario:

1. Run the engine directly — `pilot.runner.run_scenario(scenario, params, None)`
   with the scenario's documented defaults. No HTTP involved.
2. Run the same inputs through `POST /api/v1/simulations/{scenario_id}`.
3. Compare **every** returned value: each element of each named series array,
   each scalar metric, and each numeric cell of each downloadable table.

Both paths call the same function objects in the same process. The only
transformation between them is Python `float` → JSON → Python `float`.

## 2. Tolerance, and why it is what it is

**`rtol = 1e-9`, `atol = 0.0`.**

This is not a number invented for the migration. It is the repository's only
existing numerical tolerance, taken from `tests/test_regression_baseline.py:31`,
where it guards the published research results. Using anything looser here
would mean the migration was held to a weaker standard than the science it
carries.

In principle the comparison should be *exact*, and it is: Python's `repr`-based
float serialisation round-trips IEEE-754 doubles losslessly, so a value that
survives JSON is the same value. The 1e-9 band is therefore slack that was
never consumed. It is kept only so that a future change of JSON encoder cannot
fail this suite for a reason unrelated to physics. **Any real difference would
be orders of magnitude larger than the tolerance and would fail loudly.**

### 2.1 Tolerances used elsewhere, unchanged

| Check | Tolerance | Source |
| --- | --- | --- |
| Regression baseline | `rtol = 1e-9`, `atol = 0` | `tests/test_regression_baseline.py:31` |
| Energy drift (absolute guard) | `< 1e-11` | same |
| Period error (absolute guard) | `< 1e-9` | same |
| CW versus closed-form STM | `< 1e-6` m | same |
| RK4 observed order | `3.6 ≤ order ≤ 4.4` | `tests/test_integrator_convergence.py` |
| Step-independence of published metrics | relative change `< 1e-3` | same |
| Published J2 gap | 2.78 / 5.81 m ± 0.02 | activity content tests |
| Published rendezvous | 1595 s ± 25, 2.33 m/s ± 0.05, 0.017 m/s² ± 0.001 | same |

None was loosened. None was invented.

### 2.2 Treatment of missing values

JSON has no `NaN`. The engine legitimately produces one — a rendezvous that
never converges has no convergence time — so `NaN` and `±Infinity` serialise as
`null`, and the comparison treats `null` and `NaN` as the same absence. A test
asserts the finite/non-finite *pattern* matches element-for-element, so a `NaN`
appearing or disappearing is a failure rather than a coincidence.

---

## 3. Results

| Scenario | Result kind | Series | Metrics | Tables | Values compared | Max abs. diff | Max rel. diff | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `two_body_orbit` | orbit | 8 | 7 | 2 | 48,028 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `j2_effect` | orbit_comparison | 8 | 4 | 1 | 19,216 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `drag_effect` | orbit_comparison | 8 | 4 | 1 | 72,016 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `cw_natural_motion` | relative | 13 | 6 | 2 | 28,924 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `pd_rendezvous` | relative | 13 | 6 | 2 | 28,924 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `gain_tuning` | gain_sweep | 2 | 4 | 1 | 124 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `cw_vs_reference` | comparison | 27 | 9 | 4 | 47,835 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `integrator_convergence` | convergence | 3 | 3 | 1 | 51 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `proximity_safety` | collision | 18 | 4 | 3 | 37,816 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `external_validation_export` | external_export | 7 | 3 | 2 | 7,787 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| `research_workbench` | comparison | 27 | 9 | 4 | 47,835 | 0.000 × 10⁰ | 0.000 × 10⁰ | **Pass** |
| **Total** | | **134** | **59** | **23** | **338,556** | **0.000 × 10⁰** | **0.000 × 10⁰** | **11 / 11** |

---

## 4. What was compared, by category

Each of these is covered by a named test.

| Category | Covered by | Result |
| --- | --- | --- |
| **Units** | `test_defaults_and_units_match_the_catalogue` — every parameter's unit string compared against the catalogue `Param` | Pass, 11 scenarios |
| **Defaults** | Same test — `default`, `minimum`, `maximum`, `step` compared exactly | Pass, 100 parameters |
| **Derived quantities** | `test_scalar_metric_parity` — every metric the API reports | Pass, 59 metrics |
| **Time arrays** | `test_series_array_parity` — `t_s` element-wise | Pass |
| **State vectors** | Same — `x_m`…`vz_mps`, both CW and reference branches | Pass |
| **Conservation metrics** | `test_conservation_metrics_match_the_regression_baseline` | Pass |
| **Control outputs** | `test_series_array_parity` — `ax_mps2`, `ay_mps2`, `az_mps2`, `ctrl_mag_mps2`, delta-v | Pass |
| **Perturbation outputs** | `test_cw_vs_reference_reproduces_the_published_j2_figures` | Pass |
| **Validation values** | `test_pd_rendezvous_reproduces_the_published_metrics`, `test_integrator_convergence_reports_fourth_order` | Pass |
| **Error handling** | 12 tests in `test_api.py` covering every silent-wrong-answer case | Pass |

### 4.1 Published results reproduced through the HTTP API

Not merely "the two paths agree" — the API reproduces the figures the research
publishes:

| Quantity | Published | Through the API | Test |
| --- | --- | --- | --- |
| 400 km orbital period | 5553.62 s | 5553.624271252228 s | `test_conservation_metrics…` |
| Two-body energy drift | ~1e-13 | 4.58 × 10⁻¹³ (at this grid), guard `< 1e-11` | same |
| CW-vs-reference, two-body | 2.78 m | 2.78 ± 0.02 | `test_cw_vs_reference…` |
| CW-vs-reference, with J2 | 5.81 m | 5.81 ± 0.02 | same |
| J2 error ratio | "roughly doubles" | 1.8 < ratio < 2.5 | same |
| PD rendezvous convergence | ≈ 1595 s | 1595 ± 25 | `test_pd_rendezvous…` |
| PD rendezvous delta-v | 2.33 m/s | 2.33 ± 0.05 | same |
| PD peak thrust | 0.017 m/s² | 0.017 ± 0.001 | same |
| Initial separation | 369.32 m | 369.32 ± 0.01 | same |
| RK4 observed order | ~4 | 3.6 ≤ order ≤ 4.4 | `test_integrator_convergence…` |

### 4.2 Determinism

`test_identical_requests_give_identical_numbers` runs the same request twice
and asserts every series and every metric is identical. This is what makes the
reproducibility claim in every exported scenario configuration true rather than
aspirational.

---

## 5. Discrepancies found and resolved

Two, both found *by* this suite, both in the new adapter layer, neither in the
physics.

### 5.1 Duplicate metric key with two different meanings — **resolved**

`test_scalar_metric_parity` failed on `cw_vs_reference` and
`research_workbench`: the API reported `final_pos_error_m` as
2.097 × 10⁻⁹ where the engine gave 1.516 × 10⁻⁷.

Not a numerical error. The comparison scenarios emitted **`final_pos_error_m`
twice** — once from the rendezvous metrics (the chaser's distance from the
*target*) and once from the model-comparison summary (the gap between the two
*models*). Two unrelated quantities under one name; whichever came last won.

Renamed the comparison metrics to `model_gap_max_m`, `model_gap_final_m` and
`model_gap_max_vel_mps`. Added `test_metric_keys_are_unique`, which also checks
series keys, plot ids and table ids, so this class of collision cannot return.

Worth stating plainly: the engine was always right. The defect was in how the
new layer named its outputs, and it would have shown a participant the wrong
number under a plausible label.

### 5.2 Error responses rejected by the endpoint's own response model — **resolved**

`POST /simulations/{id}` declared `response_model=SimulationResult` and returned
an `ErrorResponse` on failure. FastAPI validated the response against the
declared model and turned a clean 422 into an unhandled 500. Nine tests caught
it. Now returned as an explicit `JSONResponse`, which bypasses response-model
validation.

**No unresolved discrepancy remains.**

---

## 6. Presentation parity

Separate from numerical parity, and separately tested, because a correct number
displayed as `2e-12` is a presentation defect even though the value is right.

| Check | Test | Result |
| --- | --- | --- |
| No machine notation in any interpretation sentence | `test_no_interpretation_contains_machine_notation` | Pass, 11 scenarios |
| No emoji in any user-facing string | `test_no_user_facing_text_contains_an_emoji` | Pass, 11 scenarios |
| Python and TypeScript formatters agree | `test_formatting_parity.py` + `formatting.test.ts` | Pass, shared case table |
| Exported CSV carries machine values, not display strings | `components.test.tsx` | Pass |

One leak was found here and fixed: the energy-drift interpretation used an
f-string and printed `5e-13` into a sentence a participant reads. It now uses
the shared formatter and reads `4.6 × 10⁻¹³`.

---

## 7. Runtime

Measured through the HTTP layer, at each scenario's defaults:

| Scenario | Steps | Runtime |
| --- | ---: | ---: |
| `external_validation_export` | 555 | 46 ms |
| `cw_natural_motion` | 1 111 | 63 ms |
| `pd_rendezvous` | 1 111 | 119 ms |
| `proximity_safety` | 1 111 | 170 ms |
| `two_body_orbit` | 4 000 | 178 ms |
| `j2_effect` | 4 000 | 320 ms |
| `cw_vs_reference` | 1 111 | 386 ms |
| `integrator_convergence` | fixed internal grid | 485 ms |
| `gain_tuning` | 16 × 555 | 777 ms |
| `research_workbench` | 1 111 | 1 721 ms |
| `drag_effect` | 20 000 | 2 514 ms |

HTTP and serialisation overhead is not separable from engine time at this
resolution and is not material: the dominant cost is the pure-Python
integration loop, as it was in the Streamlit version.

---

## 8. What this report does **not** claim

* It does **not** claim the engine is correct in an absolute sense. It claims
  the new API returns exactly what the existing engine returns. The engine's own
  correctness rests on its conservation checks, its agreement with closed-form
  solutions, and its measured fourth-order convergence — all unchanged, all
  still passing (255 tests).
* It does **not** constitute external validation. **No GMAT or STK comparison
  has ever been run for this repository.** The workflow to run one is exposed;
  the result does not exist. See `KNOWN_GAPS.md` §1.
* Parity was measured on one platform. A different BLAS or CPU could in
  principle produce last-bit differences in matrix products, which is exactly
  what the 1e-9 band absorbs. Cross-platform parity is untested — the CI matrix
  covering Ubuntu and Windows has not yet run.
