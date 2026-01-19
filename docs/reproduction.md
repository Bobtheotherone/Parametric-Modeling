# Reproduction Procedure (M0)

This is the canonical, deterministic reproduction path for M0.

1) Clone the repo and check out the exact commit:

```bash
git clone <repo-url>
cd formula-foundry-tri-agent
git checkout <commit-sha>
```

2) Bootstrap the environment from the lockfile:

```bash
./scripts/bootstrap_venv.sh
```

3) Activate the venv (or use explicit interpreter paths):

```bash
source .venv/bin/activate
```

4) Run the M0 smoke and repro checks (using the venv interpreter):

```bash
./.venv/bin/python -m tools.m0 smoke
./.venv/bin/python -m tools.m0 repro-check
```

5) Run the verification gates:

```bash
./.venv/bin/python -m tools.verify --strict-git
```

**Note:** All python commands above explicitly use `./.venv/bin/python` to ensure
the locked environment is used. Alternatively, run `source .venv/bin/activate`
first, then use `python -m ...` commands.

If any step fails, do not proceed; fix the reported issue and re-run the sequence.
