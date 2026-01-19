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

3) Run the M0 smoke and repro checks:

```bash
python -m tools.m0 smoke
python -m tools.m0 repro-check
```

4) Run the verification gates:

```bash
python -m tools.verify --strict-git
```

If any step fails, do not proceed; fix the reported issue and re-run the sequence.
