# Security Policy

## Scope

This project is a local-first ML demo with training scripts, local inference helpers, a FastAPI prediction endpoint, and a Streamlit UI.

## Reporting A Vulnerability

If you discover a security issue, do not open a public issue with full exploit details.

Share:

- affected file or workflow
- reproduction steps
- expected impact
- suggested mitigation if known

through a private maintainer channel first.

## Current Security Boundaries

- The repository does not ship production auth, user accounts, or multi-tenant controls.
- Model artifacts are expected to stay outside version control unless intentionally curated.
- Uploaded images should be treated as untrusted input.
- The local API and UI are for controlled demo or development use, not public internet exposure.

## Out Of Scope

- Third-party model vulnerabilities upstream of this repository
- Dataset quality issues without security impact
- Local environment setup problems that do not create a security boundary failure
