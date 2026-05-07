# ADR 0001: Agentic Engineering Baseline

Status: Accepted

## Context

The repository is an MVP for text/point-driven image element extraction. Future
development will involve multiple AI agents touching model adapters,
postprocessing, CLI behavior, tests, and documentation. The project needs a
repeatable way for agents to understand scope and verify changes without loading
large models for every edit.

## Decision

Adopt a lightweight agentic engineering baseline:

- Keep `AGENTS.md` and `CLAUDE.md` as agent entry points.
- Add `docs/agentic_engineering.md` as the workflow playbook.
- Add `docs/tasks/TEMPLATE.md` for larger task briefs.
- Add `scripts/verify.py` as the canonical local verification command.
- Keep unit tests focused on pure logic and boundary contracts that do not
  download model weights.

## Consequences

- Small edits can be verified quickly with `python scripts/verify.py --quick`.
- Behavior changes should add focused tests and pass `python scripts/verify.py`.
- Model-backed checks remain explicit and opt-in because they are expensive and
  depend on local GPU/model availability.

## Verification

Run:

```bash
python scripts/verify.py --quick
python scripts/verify.py
```
