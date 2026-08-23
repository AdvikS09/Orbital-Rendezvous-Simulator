# Participant guide

A short guide to using the Orbital Rendezvous Simulator. You can hand this out
as-is.

---

## What this is

A tool for exploring how spacecraft move when they are near each other in orbit.
It runs a real simulation in your browser — nothing to install, no programming.

It is built around a question that matters in real spaceflight: **spacecraft
guidance uses a simple, approximate model of relative motion because it is cheap
to compute. How far can you trust it?** You can measure that yourself.

## What it is not

Research and educational software. It is **not** NASA-approved, not
flight-certified, and not mission software. It has not been checked against
professional tools like GMAT or STK. It is good enough to learn from and to
explore ideas with; it is not something to fly a spacecraft on.

---

## Getting started

### 1. Your codes

You need two things:

* **A pilot code** — your teacher or coordinator will give you this. If you are
  here on your own, type `SELF`.
* **A participant code** — you make this up.

**Choosing a participant code.** Pick something memorable that is **not**
personal. `BLUE-FALCON-42`, `TARDIGRADE7`, `ORANGE-KETTLE` are all fine. Your
name or email address are not, and the form will reject anything that looks like
an email.

**Write it down.** If you come back another day and type the same code, the
platform recognises you and your earlier work counts. That is one of the things
the pilot is trying to measure, so it genuinely helps.

### 2. What gets recorded

Anonymous usage information: which simulations you ran, whether they worked, how
long they took, your quiz scores and your feedback.

**Not** recorded: your name, email address, age, IP address, or the numbers you
type into simulations. Your participant code itself is not stored either — only
a scrambled version of it that cannot be turned back into the code you chose.

The one exception: anything you type into the **feedback boxes** is saved
exactly as written, so do not put personal information there.

If you would rather nothing at all was recorded, close the page instead of
signing in. The simulator is open source and you can run it on your own computer
with no tracking whatsoever.

### 3. Pick a track

| Track | For |
| --- | --- |
| **Guided educational activities** | New to this. Three activities that build up from "what keeps a satellite up" to "how much can you trust a simple model". |
| **Professional / research exploration** | You already know the subject. Everything, full parameter control, reproducible exports. |
| **Free simulation** | Explore on your own terms. |

You can switch at any time.

---

## Using the guided activities

Each step works the same way:

1. **Read** what the step is about.
2. **Predict** what will happen. The Run button stays locked until you do.
3. **Run** the simulation.
4. **Look** at what actually happened, and see whether you were right.
5. **Read the takeaway.**

The prediction step is the whole point. Nobody records what you predict and
there is no penalty for being wrong — but committing to an answer before you
look is the difference between running an experiment and watching an animation.
You will remember the surprising results, and some of them are genuinely
surprising.

---

## Making sense of what you see

### The frames

Two different views of the same motion:

* **The orbit view** shows the spacecraft going round the Earth, seen from
  outside.
* **The relative view (LVLH)** puts the *target* spacecraft at the centre and
  shows where the chaser is relative to it. Three directions:
  * **radial (x)** — straight up, away from Earth
  * **along-track (y)** — forwards, the way the target is travelling
  * **cross-track (z)** — sideways, out of the orbit plane

The relative view is where the surprises live, because that frame is rotating.

### Numbers you will meet

| Term | What it means |
| --- | --- |
| **Δv (delta-v)** | Total change in velocity the engine had to produce. It converts directly into how much propellant you must carry, so it is the price of a manoeuvre. |
| **Convergence time** | How long until the chaser got within 1 m of the target and *stayed* there. A dash means it never did. |
| **Energy drift** | How much the simulation's total energy wandered. Physics says zero, so a tiny number here means the maths is being done accurately. |
| **Perturbation** | Anything pulling the spacecraft off a perfect orbit — Earth's bulge (J2), air resistance (drag). |
| **Keep-out zone** | A protected sphere around a spacecraft that a visitor may not enter without permission. |

### Warnings

Yellow warnings do not mean something broke. They mean *read this result
carefully* — usually that you have asked for something outside the range where
the simple model is reliable, or that your time step is too coarse. They are
worth reading; they usually explain the odd thing you were about to notice.

Red messages mean the run did not happen, and say what to change.

---

## Downloading your work

After any simulation you can download:

* **CSV files** — the numbers, for a spreadsheet or your own plots.
* **Scenario configuration (JSON)** — everything needed to reproduce that exact
  run later. The simulator has no randomness, so the same settings give the same
  answer every time.
* **Notes and caveats (TXT)** — the assumptions and limitations behind the
  result. Keep it with the CSVs; a spreadsheet of numbers with no context is how
  results get misread later.

Nothing in any download identifies you.

---

## Things worth trying

* Put the chaser **1 km behind** the target with no thrust. Then put it **100 m
  above**. One of these barely moves and the other travels kilometres. Work out
  why.
* In the rendezvous scenario, raise the controller bandwidth and watch the Δv.
  Then ask which setting you would actually choose.
* Turn Earth's bulge (J2) on and off in the model-comparison scenario. It
  roughly doubles the error.
* In the same scenario, turn the **controller on** and see the difference almost
  vanish. Then think about why that makes a closed-loop test a bad way to judge
  a model.

---

## At the end

There is a short feedback form. It genuinely matters — this is a pilot, and the
point of a pilot is to change what gets built. Bug reports are the most useful
thing you can leave.

Your feedback is anonymous, which means we cannot write back to you about it.

---

## If something goes wrong

| Problem | Try this |
| --- | --- |
| A red error message | Read it — it says which value is out of range. Change it and run again. |
| "This run needs too many steps" | Increase the timestep or shorten the duration. The shared server has limits your own computer would not. |
| "Too many simulations in under a minute" | Wait a moment. The server is shared with everyone else. |
| The page is stuck or blank | Reload. You will need to sign in again — use the **same participant code**. |
| The plot looks empty | Check the warnings. Usually the numbers are all much smaller than the axis range. |
| It is slow | Big simulations take a few seconds. If it is worse than that, the server may be busy. |
