# Deep Dive

Study notes, Q&As, and code walkthroughs accumulated while reading TraceML in
depth. These pages sit alongside the User Guide and Developer Guide but are
not part of the official documentation surface — they are reading aids and
working notes, written for someone studying the codebase or onboarding to
the architecture.

## What's here

- **[Why TraceML](why.md)** — the value thesis: what problem TraceML solves,
  who it's for, and how it differs from existing profilers.
- **[Learning Q&A](learning-qa.md)** — questions that came up while reading the
  code, with answers grounded in the current source.
- **[Code walkthroughs](code-walkthroughs.md)** — long-form tours through
  specific subsystems, tracing the actual call paths.
- **[PyTorch Q&A](pytorch-qa.md)** — PyTorch internals relevant to TraceML's
  instrumentation strategy (hooks, autograd, CUDA timing, DDP).

## How to read these

These pages reference specific files and line numbers in TraceML at a point
in time. Where a note says "see `samplers/step_time_sampler.py`", the file
may have moved or changed since the note was written — treat the structural
explanations as durable, but verify symbol names and line numbers against
the current source before relying on them.
