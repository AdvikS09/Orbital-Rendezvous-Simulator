# UI migration report

How the Streamlit presentation layer was replaced, and what changed for the
person using it.

---

## 1. The two defects this migration was asked to fix

### 1.1 Emoji

The Streamlit interface used **59 emoji across 19 distinct glyphs** as status
markers and section decoration.

| Glyph | Count | Replaced by |
| --- | ---: | --- |
| ⚠ | 13 | `warning` icon |
| ✅ | 10 | `success` icon |
| 🚫 | 8 | `error` icon |
| 🛰 | 4 | `orbit` icon |
| 🚨 | 4 | `error` icon |
| ⬇ | 3 | `download` icon |
| ▶ | 2 | `play` icon |
| 🔬 | 2 | `flask` icon |
| 🔒 | 2 | `lock` icon |
| ℹ | 2 | `info` icon |
| 🎓 📐 📈 ⚖ 🔍 💡 🎉 🧪 👋 | 1 each | `book`, `ruler`, `chart`, `chart`, `target`, `info`, `success`, `flask`, `user` |

Replaced by 26 line icons on a 24-unit grid, 1.5px stroke, inheriting
`currentColor`. They are `aria-hidden` by default because each sits beside a
text label that already carries the meaning — a screen reader previously
announced 📐 as "triangular ruler", which is noise rather than information.

**Verified zero.** A `\p{Extended_Pictographic}` scan finds none in
`web-platform/`, and two test suites enforce it: one over every API string for
every scenario, one over rendered component output.

Mathematical characters are **kept**, being notation rather than decoration:
`Δ ² × → · — – ≈ − §` and the superscript digits scientific notation needs.

### 1.2 Machine notation

The Streamlit interface rendered values with Python's `%g`, which put strings
like `2e-12` and `1.1060555202205516e-13` on screen.

`web-platform/frontend/src/formatting/number.ts` is now the single formatting
utility. `2e-12` renders as **`2.0 × 10⁻¹²`** with real Unicode superscripts,
a proper multiplication sign, and a true minus (U+2212) that aligns with digits.

It supports every mode the brief asked for:

| Mode | Function | Example |
| --- | --- | --- |
| Fixed decimals | `formatFixed` | `5553.62` |
| Significant figures | `formatSignificant` | `2.326` |
| Scientific | `formatScientific` | `1.11 × 10⁻¹³` |
| Automatic | `formatNumber` | switches outside 10⁻³ … 10⁵ |
| Percentage | `formatPercent` | `50.0%` |
| Angle | `formatAngle` | `51.60°` |
| Duration | `formatDuration` | `92.6 min` |
| Distance | `formatDistance` | `369.3 m`, `50 km` |
| Velocity | `formatVelocity` | `2.326 m/s` |
| Acceleration | `formatAcceleration` | `0.01702 m/s²` |
| **Export values** | `rawValue` | `1.1060555202205516e-13` |

`rawValue` is the important one. Presentation formatting must never reach a
CSV, because someone will load it into a spreadsheet and compute with it. A
test asserts `rawValue` output never contains a superscript, a multiplication
sign or a Unicode minus, and round-trips through `Number()` without loss.

**One leak was found by testing in the browser and fixed:** backend-composed
interpretation sentences still used an f-string, so the energy-drift reading
printed `5e-13` mid-sentence. It now uses the shared server-side formatter and
reads `4.6 × 10⁻¹³`. A guard test scans every interpretation of every scenario.

---

## 2. Architecture change

| | Streamlit | React + FastAPI |
| --- | --- | --- |
| Rendering | Server, full script re-run per interaction | Client, React reconciliation |
| Plots | Built in Python, serialised as figure JSON | Built in the browser from numeric arrays |
| State | `st.session_state`, string-keyed | React context + component state |
| Routing | A stage string in session state | React Router, real URLs |
| Admin | `?view=admin`, then a password form | HTTP Basic on every admin endpoint |
| Physics | `src/` | `orbital_core/`, **unchanged** |

The frontend never recomputes a trajectory. It receives named numeric arrays
plus plot *metadata* — axis titles, units, which series belongs on which axis,
which colour index and dash pattern — and draws exactly that.

### 2.1 Deep links

Every view now has a URL. `/scenarios/pd_rendezvous` is shareable and
bookmarkable; the Streamlit version had one URL for the whole application.

---

## 3. Visual direction

Deep navy and graphite surfaces, restrained cyan accent, no decorative
gradients, no oversized headings. 115 design tokens in
`src/styles/tokens.css`, in dark and light themes.

Typographic decisions that matter for a numbers application:

* **Tabular numerals everywhere** (`font-variant-numeric: tabular-nums` on
  `body`), so digits align between rows of a results table.
* **Numeric columns right-aligned**, so decimal points line up.
* **Units in their own element**, separated by a non-breaking space so a value
  can never wrap away from its unit.
* **The largest type size in the entire application is 22px.** There is no
  hero heading.

### 3.1 Colour is never the only signal

* Status callouts carry a visually-hidden text prefix ("Warning:", "Error:")
  before the message, so severity survives greyscale printing and screen
  readers.
* The active navigation item has an inset rule as well as a tint.
* Plot series are distinguished by **dash pattern as well as hue**, and use the
  Okabe-Ito palette carried over unchanged from the Streamlit version — so a
  series keeps both its colour and its line style across scenarios.

---

## 4. Numeric input

The brief called this out specifically. Each requirement, and how it is met:

| Requirement | Implementation |
| --- | --- |
| Sliders **and** editable numeric fields | `SliderField` renders both, synchronised |
| Unit beside the field | A suffix inside the input shell, a flex sibling — never an overlay |
| Reasonable decimal precision | Per-parameter `fmt`/`precision` from the API; gains use scientific notation because `0.000046` is unreadable |
| Values never overlap labels or handles | The slider's value sits in its own right-aligned cell **above** the track. A test asserts the value element is not a descendant of the range input |
| Tabular numerals | `.num` class, applied by `NumberValue` |
| Right-aligned numerics | `.num--right`, used in every results table |
| Scientific notation for tiny values | Automatic below 10⁻³ |

Two behaviours worth naming:

* **Scientific input is accepted.** Typing `4.6e-5` into a gain field works —
  the field is `type="text"` with `inputMode="decimal"` rather than
  `type="number"`, which mangles exponent entry in several browsers.
* **Nothing is silently clamped.** An out-of-range value is reported with the
  server's own explanation. Clamping would recreate exactly the
  silent-wrong-answer problem the validation layer exists to prevent.

---

## 5. Plots

All eight result kinds render through Plotly.js from returned arrays.

| Streamlit builder | New | Notes |
| --- | --- | --- |
| `orbit_3d` | `orbit3d` | Earth to scale; metres converted to km for display only |
| `lvlh_trajectory` | `scatter3d` | Target marker at the origin |
| `series` / `components` | `line` | |
| `control_history` | `line` | The `\|u\|` trace is marked explicitly rather than by hard-coded index |
| `dual_error` | `dual_axis` | |
| `gain_heatmap` | `heatmap` | |
| `proximity_3d` | `scatter3d` | Translucent keep-out sphere |
| `convergence_plot` | `log_log` | With the computed fourth-order reference line |

Every axis carries its unit. Legends, hover values and PNG download are
preserved. No smoothing is applied anywhere — the line through the points is
the data.

**Lazy loading.** Plotly is 4.67 MB. It sits in its own chunk behind a dynamic
`import()`, so the initial bundle is **21 kB** and the library loads only when
a result is first rendered. On failure the chart is replaced by a callout
saying the numbers are unaffected and still downloadable, rather than crashing
the page.

---

## 6. Accessibility

| Area | Status |
| --- | --- |
| Labels | All 19 form controls have an associated `<label htmlFor>` or `aria-label`. Verified mechanically |
| Errors | `role="alert"`, `aria-invalid`, `aria-describedby` on every field |
| Icon-only buttons | All four have `aria-label` |
| Focus | Visible 2px ring on every interactive element |
| Skip link | Present |
| Tabs | Full ARIA tabs with roving tabindex and arrow-key navigation |
| Scroll regions | Wide tables are `role="region"` with `tabIndex={0}` so keyboard users can scroll them (WCAG 2.1.1) |
| Reduced motion | Transitions disabled; the spinner falls back to a stepped rotation rather than vanishing |
| Lists | `role="list"` retained on styled lists, because `list-style: none` strips list semantics in Safari/VoiceOver |

ESLint runs `jsx-a11y` with those rules as **errors**. Two rules are configured
rather than disabled, with the reasoning recorded in `.eslintrc.cjs`.

**Not verified:** actual screen-reader testing, and measured contrast ratios.
See `KNOWN_GAPS.md`.

---

## 7. Responsive layout

| Width | Layout |
| --- | --- |
| ≥ 900px | Persistent sidebar, parameter panel beside the results |
| < 900px | Sidebar collapses to a drawer behind a toggle; panels stack, parameters first |
| < 480px | Tighter padding |

Verified at 375 × 812: **zero horizontal page overflow.** Wide tables scroll
inside their own container.

**A real bug was found and fixed here.** Resizing from desktop to mobile with
charts already drawn produced 426px of overflow: Plotly writes an explicit pixel
width onto its root element, and that oversized child propped its container
open, so the `ResizeObserver` never observed a size change and never requested
the resize that would have shrunk the chart — a deadlock. Adding `min-width: 0`
and `overflow: hidden` to the frame lets it shrink, which fires the observer.
Overflow is now zero and the chart reflows from 760px to 301px. This matters
for a tablet rotating to portrait.

---

## 8. Content migration

All substantive wording was carried across **verbatim**, minus emoji:

| Content | Where it now lives |
| --- | --- |
| Scenario summaries, explanations, reading guides | Served by `GET /scenarios/{id}` from the same catalogue objects |
| Assumptions and limitations | Same; shown beside every result and written into every export |
| Landing, consent, sign-in, confirmation copy | `src/pages/*`, and the two status statements are imported from one module so they cannot drift |
| Activity briefs, predictions, "look for", takeaways | `GET /activities` |
| Assessment questions and explanations | `GET /assessments/{id}` |
| Validation error and warning messages | Produced by `pilot/validation.py`, unchanged |
| Result interpretations | Ported with their branching logic intact |

Scenario content is **not** duplicated into TypeScript. It is served from the
Python catalogue that the Streamlit application also reads, which is stronger
than the brief's "structured JSON or TypeScript data": there is exactly one
copy, so the two frontends cannot disagree.

---

## 9. What a user gains, and what they lose

**Gains**

* Numbers that read as numbers — `2.0 × 10⁻¹²`, not `2e-12`.
* No full-page re-run on every widget interaction.
* Shareable URLs.
* Faster interaction: parameter changes are local, only Run costs a round-trip.
* A preflight cost estimate before committing to a run.
* Client-side CSV generation, so downloads are instant.
* A genuinely protected admin API rather than an unadvertised URL.

**Losses**

* Requires JavaScript. The Streamlit version did too, so this is not a
  regression, but neither is a no-JS fallback provided. The engine remains
  runnable from the command line.
* A build step now exists. `npm run build` must run before deployment; the
  Dockerfile does it.
* The initial page load fetches a bundle rather than server-rendered HTML.
  At 21 kB plus 164 kB of React, this is small — but Plotly's 4.67 MB arrives
  on first chart render.

---

## 10. Files

**Frontend** — 89 files under `web-platform/frontend/`:

* `src/formatting/` — the number formatter
* `src/api/` — types and client
* `src/components/ui/` — 9 primitives
* `src/components/controls/` — 6 input components
* `src/components/results/` — 8 result components
* `src/components/admin/` — 2 reporting components
* `src/charts/` — Plotly layer, 5 modules
* `src/pages/` — 12 pages
* `src/state/` — session context
* `src/styles/` — tokens and global

**Tests** — 77: 44 formatting, 24 component, 9 chart-trace.

**Not removed:** every Streamlit file. `pilot/ui/`, `pilot_app.py` and `app.py`
all still run.
