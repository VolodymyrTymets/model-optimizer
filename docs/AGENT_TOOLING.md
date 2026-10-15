# Agent tooling and MCP policy

> The single source of truth for what the agent may reach in this engagement. Part of the client onboarding conversation.

## Client consent
- AI-assisted development permitted: yes (own repository) — recorded 2026-10-15, by the maintainer
- Restrictions: recorded EMG/audio/MRI data under `assets/` never enters a prompt, log or PR (rule G1)

## Active MCP servers
| Server | Purpose | Data it can reach | Scope/permissions | Approved by | Rotate/expire |
|--------|---------|-------------------|-------------------|-------------|---------------|
| context7 | library docs | none (library names only) | read | internal | — |
| serena | code navigation | this repo, locally | read | internal | — |
| github | PRs, issues, CI | this repo | repo-scoped token, no admin/delete | maintainer | on token expiry |

## Forbidden here
- Production databases, production credentials, production deploys (rule C5)
- Any server not listed above
- Shared service accounts for OAuth servers — each developer authenticates as themselves

## Notes
- Keep ≤5 active servers per session; disable what the current task does not need.
- Prefer an already-installed CLI over an MCP server when both do the job — a CLI costs no context.
- Secrets only via `${ENV_VAR}` in `.mcp.json`. Never inline.
