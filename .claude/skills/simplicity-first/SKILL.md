---
name: simplicity-first
description: Simplicity gate for ALL code written, fixed, or reviewed in this repo. Invoke BEFORE writing any code change and when reviewing any diff. Treats over-complex or oversized code as a correctness bug, not a style issue.
---

# Simplicity is the first principle

**The gold standard is: a human can read it and understand it.** Every rule
below is an instance of that one test. These are hard correctness rules, not
style preferences. A violation is a bug and must be fixed before the change
ships.

1. **Code a human can't follow at a glance is a bug.** If a reviewer can't read
   a function top-to-bottom in one pass, restructure or delete it. Nesting,
   indirection, and clever constructs count against correctness — cleverness
   that costs comprehension is a defect, whatever it saves.

2. **Too much / redundant code is a bug.** Solve the problem in the fewest lines
   that stay readable. Prefer deleting code over adding it. A fix that adds more
   than ~20 lines for a problem statable in one sentence is suspect — find the
   smaller fix first.

3. **Simplicity is the core engineering metric.** When two designs both work,
   ship the one with less code, fewer concepts, fewer files. Never add config,
   record types, or return-shape changes "for the future".

4. **No over-encapsulation.** No new class / dataclass / helper / module for a
   single call site. A helper needs 3+ real call sites AND nontrivial logic —
   otherwise inline it. Never wrap trivial code. Never change a function
   signature or return shape to thread data that only one caller needs.

## Checklist before finishing any change

- Would a human reading this cold understand it in one pass? That is the gate.
- Could this diff be half the size? If unsure, make it smaller.
- Any new class or file? Justify each with 3+ call sites, or delete it.
- Any signature / return-shape change? Verify every caller genuinely needs it.
- Comments: concise "why" only, 2-4 lines max, release quality.
- One problem = one minimal diff. Do not batch unrelated "improvements".
- A "bug" that cannot trigger under the real recipes is not worth fixing.
