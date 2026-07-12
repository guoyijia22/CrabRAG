# Changelog

## 1.3.1 - 2026-07-12

- Replaced the duplicate full index-governance panel on the Knowledge Base page with a compact active-generation, document-count, and warning-count summary.
- Added an in-app link from the compact summary to the dedicated Index Governance page while retaining its complete status and rollback controls.
- Clarified that foreground sessions should stop with `Ctrl+C`, while `stop.bat` remains the verified administrative stop path for background, detached, recovery, backup, and restore workflows.

## 1.3.0 - 2026-07-12

- Added replaceable OIDC/JWT identity verification with issuer, audience, signature, algorithm, expiry, not-before, and `kid` validation.
- Added a fail-closed enterprise HTTP permission adapter and trusted internal-token rotation with a bounded previous-token grace period.
- Added append-only, hash-chained security auditing with an atomic anchor and the `audit-verify` administration command.
- Moved model API credentials to environment variables or the operating-system keyring; legacy plaintext JSON credentials are migrated transactionally.
- Protected application, retrieval, model, and sidebar settings with trusted administrator authorization.
- Preserved existing HTTP payload models, manifests, Chroma collections, generations, and single-machine deployment behavior.

## 1.2.0 - 2026-07-12

- Added versioned fixed evaluation datasets, nine quality metrics, quality gates, and generation/configuration-bound approvals.
- Added gated Dynamic Top-K, parent context, and near-duplicate evidence removal.
- Added stable citation identities and real ACL/inactive-content leakage checks.

## 1.1.0 - 2026-07-11

- Restored maintainable React/TypeScript/Vite and Bun/Hono sources with reproducible generated artifacts.
- Added Windows/Ubuntu CI, doctor/backup/restore commands, bounded runtime memory, and verified Windows release packaging.
