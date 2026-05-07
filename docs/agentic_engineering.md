# Agentic Engineering Playbook

This project should be easy for a fresh AI agent to inspect, change, verify,
and hand off without relying on hidden chat context. Keep every change anchored
to a small, testable contract.

## Source Of Truth

- `AGENTS.md`: repository-specific operating rules for Codex-style agents.
- `CLAUDE.md`: same guidance for Claude Code-style agents.
- `docs/project_memory.md`: durable project memory and current handoff state.
- `README.md`: user-facing setup and CLI usage.
- `configs/config.yaml`: runtime behavior and model thresholds.
- `docs/adr/`: decisions that affect future design constraints.
- `docs/tasks/`: task briefs for non-trivial work.

## Standard Agent Loop

1. Read `AGENTS.md`, this playbook, and the files directly involved in the task.
2. State assumptions and success criteria before large edits.
3. Make the smallest change that satisfies the task.
4. Add or update focused tests for pure logic and integration boundaries.
5. Run an appropriate verification tier.
6. Hand off with changed files, commands run, risks, and next steps.

## Verification Tiers

- Tier 0, syntax and import smoke:
  `python scripts/verify.py --quick`
- Tier 1, pure unit tests:
  `python scripts/verify.py`
- Tier 2, CLI sanity without model downloads:
  `python main.py --help` and `python main.py info -i <local-image>`
- Tier 3, model-backed quality check:
  run one `extract-text` or `extract-point` command against a known image after
  confirming model weights are already available or network access is intended.

Use Tier 0 before handing off any Python edit. Use Tier 1 for behavior changes.
Use Tier 3 only when changing model calls, prompt conversion, postprocessing, or
output quality.

## Engineering Boundaries

- Internal images are `numpy.ndarray` values with shape `(H, W, 3)`, RGB,
  `uint8`.
- Masks passed to OpenCV must be squeezed to 2D `(H, W)` before morphology.
- Model classes keep explicit `load()` and `unload()` lifecycles.
- Runtime choices belong in `configs/config.yaml`, not hardcoded call sites.
- Heavy model tests should not be required for normal CI or quick agent checks.

## Task Brief Template

For non-trivial work, create a copy of `docs/tasks/TEMPLATE.md` and fill in:

- Goal and user value.
- Files likely in scope.
- Out-of-scope boundaries.
- Acceptance checks.
- Verification commands.
- Handoff notes.

Short one-file fixes do not need a task brief, but the final answer should still
include the verification performed.

## Security And Local State

- Keep API keys, auth tokens, and local tool configs out of source control.
- `.codex/config.toml`, `.codex/settings.local.json`, and
  `.claude/settings.local.json` are local-only files.
- Do not commit `outputs/`, `my_outputs/`, model checkpoints, caches, or virtual
  environments.

## Decision Records

When a change affects architecture, model backend contracts, output formats, or
agent workflow, add an ADR under `docs/adr/` using a short numbered filename.
Keep ADRs concise: context, decision, consequences, and verification.
