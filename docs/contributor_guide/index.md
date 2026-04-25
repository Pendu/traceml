# Contributor Guides — TraceML

> Internal index. Audience: one trusted co-founder / new engineer being onboarded to TraceML. Not for public docs.

This folder is the playbook for extending TraceML. Each guide covers one feature type end-to-end — when it applies, what files you touch, what conventions you follow, what to check before opening a PR. Reviewer guides cover the same axis from the other side: what to verify before approving.

**Last verified:** 2026-04-25.

---

## Start here

Before opening an editor, read these two foundation docs:

- **[principles.md](principles.md)** — cross-cutting rules every change must obey. Fail-open, overhead budget, wire compatibility, logging, versioning, smoke-test discipline. Every per-feature guide refers back here instead of restating the rules.
- **[pipeline_walkthrough.md](pipeline_walkthrough.md)** — the end-to-end data flow (sampler → buffer → queue → DB → sender → TCP → aggregator → store → renderer → driver). Condensed from PR #87 Appendix D. If you can't trace your feature through this pipeline, you don't understand it well enough yet.

The full pedagogical walkthrough is in [W6–W11](../deep_dive/code-walkthroughs.md) of the learning archive — read those once if you haven't.

---

## Decision tree — which guide do I need?

```
What are you trying to do?

├── Measure something new about training
│   │
│   ├── Is the data already collected by an existing sampler?
│   │   ├── Yes → you probably need a renderer or diagnostic, not a sampler
│   │   │   ├── Visualize it → add_renderer.md
│   │   │   └── Surface an opinionated finding from it → add_diagnostic.md
│   │   │
│   │   └── No → you need either a sampler or a patch
│   │       ├── Polling a passive source (psutil, pynvml, /proc) → add_sampler.md
│   │       └── Hooking a PyTorch op (timing, memory, inputs) → add_patch.md
│   │
│   └── If the new sampler emits rows the renderer needs windowed,
│       you also need → add_sqlite_projection.md
│
├── Visualize existing data differently
│   └── add_renderer.md
│
├── Surface an opinionated finding (INPUT-BOUND, MEMORY-CREEP, IDLE-GPU, ...)
│   └── add_diagnostic.md
│
├── Add a CLI option, flag, --mode, or environment variable
│   └── add_cli.md
│
├── Support a new framework (HuggingFace, Lightning, JAX, ...)
│   └── add_integration.md
│
└── Change the wire format / schema
    └── change_wire_format.md  (high-risk — talk to Abhinav first)
```

---

## Index — author guides

Each guide tells you: when this is the right pattern, the contract you're honoring, the files you'll touch, the decisions to make, the pitfalls to avoid, and the checklist for a high-quality PR.

| Guide | Feature type | Risk | Reference PR |
|---|---|---|---|
| [add_sampler.md](add_sampler.md) | New sampler / metric | Medium | — |
| [add_renderer.md](add_renderer.md) | New display view (CLI / NiceGUI) | Low | — |
| [add_patch.md](add_patch.md) | New instrumentation patch / hook | High (PyTorch coupling) | **#87 (h2d timing)** |
| [add_diagnostic.md](add_diagnostic.md) | New diagnostic verdict | Low–Medium | — |
| [add_cli.md](add_cli.md) | New CLI command, flag, or `--mode` | Low | — |
| [add_sqlite_projection.md](add_sqlite_projection.md) | New SQLite projection writer | Medium | — |
| [add_integration.md](add_integration.md) | New framework adapter (HF / Lightning / JAX / ...) | Low–Medium | — |
| [change_wire_format.md](change_wire_format.md) | Wire format / schema migration | **High** (every v0.2.x user is a stakeholder) | — |

---

## Index — reviewer guides

Reviewer guides are workflow playbooks: anchor the PR to walkthroughs, run the consistency check, apply class-specific failure-mode questions, write verification gates, draft comments, land the verdict. Each reviewer guide reuses the seven-step meta-review-workflow first introduced in `review_patch.md`.

| Guide | Companion author guide |
|---|---|
| [review_patch.md](review_patch.md) | add_patch.md |
| [review_sampler.md](review_sampler.md) | add_sampler.md |
| [review_renderer.md](review_renderer.md) | add_renderer.md |
| [review_diagnostic.md](review_diagnostic.md) | add_diagnostic.md |
| [review_cli.md](review_cli.md) | add_cli.md |
| [review_sqlite_projection.md](review_sqlite_projection.md) | add_sqlite_projection.md |

A reviewer guide for `add_integration.md` and `change_wire_format.md` is not yet written — for now use the author guide's PR checklist as the verification gate.

---

## Conventions across all guides

Each author guide follows the same structure (so you can navigate predictably): intro and mental model → decisions to make → anatomy of an existing exemplar → step-by-step walkthrough of a hypothetical new feature → common patterns and exemplars → schema/contract rules → overhead budget → testing → common pitfalls → PR checklist → appendix → gaps and ambiguities.

Each guide opens with a metadata block — feature type, risk level, cross-cutting impact, PyTorch coupling, reference PRs, companion reviewer guide, last-verified date — so you can self-assess stakes before diving in.

The voice is imperative and opinionated. "Subclass `BaseSampler`. Pass two strings." Not "you might consider subclassing." If the codebase has a house style, the guide enforces it; if it doesn't, the guide says so explicitly in the gaps section.

---

## How to keep these guides current

- Each guide carries a `Last verified:` date in its metadata block. When you touch the underlying subsystem, update the date.
- The "Gaps and ambiguities" section at the end of each guide is the auditable surface — if the codebase resolves a gap, edit the section.
- Trigger conditions: if the PyTorch surfaces a guide depends on change shape (especially `_call_impl`, hook semantics, or CUDA event APIs), the guide needs revisiting. See [P51](../deep_dive/pytorch-qa.md#p51-which-torchcuda-apis-does-traceml-rely-on-and-how-stable-are-they-across-pytorch-versions-relevant-to-the-pytorch-coupling-constraint-in-claudemd) for the version contract.
- Quarterly re-verification cadence is the floor, not the ceiling.

---

## Cross-references

- **Project rules:** [../../../CLAUDE.md](../../../CLAUDE.md) (repo root) and [../../CLAUDE.md](../../CLAUDE.md) (traceml package).
- **Learning archive:** [traceml_learning_qa.md](../deep_dive/learning-qa.md), [traceml_pytorch_qa.md](../deep_dive/pytorch-qa.md), [traceml_learning_code_walkthroughs.md](../deep_dive/code-walkthroughs.md).
- **Why this product exists:** [traceml_why.md](../deep_dive/why.md).
- **PR #87 reference review:** [Notes/PR_87_review_through_walkthroughs.md](../deep_dive/pr_reviews/pr-87-h2d-timing.md). The single most important reference for `add_patch.md` and `review_patch.md`.
