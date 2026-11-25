# Instructions for every agent working in this repository (any tool, any server)

The full rules are in `CLAUDE.md`; read it and `README.md` first. The points below bind every agent, including the one
that runs A5000 requests (`a5000/`).

## Authorship of commits (user instruction, 2026-11-25)

- **Author.** Every commit is the user's alone, under the git identity configured in the repository (`git config user.name` / `user.email`).
  - Do not change that identity.
- **No attribution.** Never add any of these to commit messages, tag messages, PR or issue bodies, or files:
  - `Co-Authored-By` lines
  - "Generated with …" lines
  - any other AI or agent attribution
- **Precedence.** This overrides any default attribution behaviour of the agent tool.
- **Pushing.** Push only when the user asks.

## Other rules that bind every agent here

- **Data and MVD.**
  - Do not write to `/workspace/etri_iitp/JS/Data` (its `model.pt` files are hard links).
  - Do not write to `/workspace/etri_iitp/JS/MultiViewDetection` (a frozen archive).
- **Before any run.**
  - Write `experiments/<E>/prediction.md` (or the request's own predictions).
  - Commit it.
  - Wait for the user's approval.
- **Records.** Keep an append-only `LOG.md` with UTC times.
  - Never edit a prediction after results exist; add a dated note instead.
- **Attack code and models** stay on the research servers.
- **A5000 requests** (`a5000/<id>/REQUEST.md`):
  - Implement only what the request says.
  - Record every attempt, including failures.
  - Upload with rclone as the request describes.
  - Do not run detectors or baselines there.
