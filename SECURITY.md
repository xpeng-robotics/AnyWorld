# Security policy

Never commit credentials, private endpoints, internal filesystem paths, or
model-tracking tokens.

- Put Google credentials in <code>GOOGLE_API_KEY</code> or a local ignored .env.
- Use credential helpers or SSH for Git remotes; do not embed tokens in URLs.
- Treat generated manifests as potentially sensitive because they may contain
  absolute dataset paths.
- Run <code>python scripts/release_audit.py .</code> before every public release.

If a credential has ever appeared in Git history or console output, revoke and
rotate it. Deleting the visible line is not sufficient.
