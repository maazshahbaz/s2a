# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI routers and request/response schemas.
- `db_services/`: database access and auth helpers (Prisma-backed).
- `intelligent_pipeline/`: ASR/diarization/analysis pipeline orchestration.
- `tests/`: pytest suite plus test audio fixtures in `tests/test_audio/`.
- `api_portal/`: Next.js portal (App Router) for auth and dashboard flows.
- `sdk/javascript/` and `sdk/python/`: client SDKs.
- `prisma/`: schema and migrations.
- Root entrypoint is `main.py`; container/dev orchestration lives in `docker-compose*.yml`.

## Build, Test, and Development Commands
- Backend setup: `pip install -r requirements.txt`
- Run API locally: `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001`
- Run tests: `pytest -c tests/pytest.ini tests/`
- Fast test pass (no slow/GPU): `pytest -m "not slow and not gpu" tests/`
- E2E pipeline check: `python tests/test_e2e_pipeline.py --audio tests/test_audio/<file>.wav`
- Prisma client/migrations: `npx prisma generate` and `npx prisma migrate deploy`
- Dev stack (API + DB + portal): `docker compose -f docker-compose.dev.yml up --build`
- Portal dev (from `api_portal/`): `npm ci && npm run dev`
- JS SDK build/test (from `sdk/javascript/`): `npm ci && npm run build && npm test`

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, add type hints on new/changed code.
- JavaScript/TypeScript: follow package-local style (`api_portal` uses Next ESLint; `sdk/javascript` uses TypeScript + ESLint).
- Run quality checks before PRs: `black .`, `flake8`, `mypy .`, plus `npm run lint` in changed JS packages.
- Name tests as `test_<behavior>.py`; keep markers explicit (`@pytest.mark.integration`, `@pytest.mark.slow`, etc.).

## Testing Guidelines
- Framework: `pytest` with markers defined in `tests/pytest.ini` (`unit`, `integration`, `slow`, `gpu`, `validation`).
- Keep unit tests deterministic and isolated; mock external services (Triton/LLM/webhooks) where possible.
- For pipeline or model-dependent flows, use targeted integration tests and document required services in the PR.

## Commit & Pull Request Guidelines
- Recent history shows ticket-based and fix-style subjects (examples: `STA-186: ...`, `Fix: ...`).
- Prefer: `<ticket>: <imperative summary>` (example: `STA-199: add task generation fallback`).
- Keep commits focused; avoid vague messages like `updated prompt`.
- PRs should include: purpose, linked ticket/issue, test evidence (commands + results), config/env changes, and screenshots for `api_portal` UI updates.

## Security & Configuration Tips
- Start from `.env.example`; keep secrets in local env files, never in source control.
- Validate callback/auth changes carefully (`webhook.py`, `db_services/auth.py`) and include negative-path tests for permission or signature failures.
