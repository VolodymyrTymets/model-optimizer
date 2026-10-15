---
description: Language-neutral testing rules for model-optimizer. Python mechanics are in testing-python.md.
paths:
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/tests/**"
---

# Testing rules

This repo has **no automated test suite yet** (`unit tests: none` in the command map). Until one exists:

1. A change is verified by running the affected entry script or service on a small local data set and reporting the real output (rule B1, B2). Anything that needs GPU, Postgres or the full `assets/` data goes under "Not verified".
2. When you add the first tests, use pytest as `testing-python.md` describes, put them in `tests/`, and add the command to the command map in `CLAUDE.md` in the same PR (rule T1).
3. Mock only boundaries: Postgres (`DBClient`), the filesystem under `assets/`, TensorFlow training. Never mock the unit under test.
4. Seed every random source (`numpy`, `random`, `tf.random.set_seed`); a test never reads real recordings from `assets/` (rule G1) — build tiny synthetic signals instead.
5. No wall-clock sleeps; no test depends on another test's leftovers or order.
6. Never skip, xfail or delete an assertion to make a run green without a human-approved `# WHY:` (rule B3).
7. A bug fix starts with a test that fails for the right reason, then passes.
