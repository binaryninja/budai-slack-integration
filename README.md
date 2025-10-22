# BudAI Slack Integration

FastAPI microservice that encapsulates Slack-specific operations for BudAI.
It manages call sessions, message delivery, and App Home publishing so other
components can remain Slack-agnostic.

## Endpoints

- `GET /` – service metadata
- `GET /health` – liveness report (non-blocking dependencies)
- `POST /calls/create` – create (and persist) a call session
- `GET /calls/{call_id}` – fetch call session state
- `POST /calls/{call_id}/end` – mark a call session as finished
- `POST /messages` – post a message (text + optional blocks)
- `POST /app-home` – update the Slack App Home view

All write endpoints support optional Slack signature verification when
`SLACK_SIGNING_SECRET` is configured.

## Environment Variables

See `.env.example`. Minimum required:

- `BUDAI_OPENAI_API_KEY` (for shared tooling compatibility)
- `BUDAI_REDIS_URL`
- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`

## Local Development

```bash
uv pip install --system -r requirements.txt
uvicorn service.service:app --reload --port 8006
```

## Deployment

The included installer (`service/installer.py`) integrates with the
`budai-deploy` CLI. Railway expects a `railway.json` (see template in
`templates/`).
