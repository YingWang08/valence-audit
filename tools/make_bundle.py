"""Make notebook_bundle.zip: the committed code, for upload to the GPU notebook.

    python -m tools.make_bundle

Uses `git archive` on HEAD, so the notebook runs exactly the committed code. Refuses if code or
configuration has uncommitted changes (commit and push first). The zip contains the folder
valence-audit/ (code, config, tools, tests; no data and no .env) and valence-audit/BUNDLE_INFO.json
with the commit id; the git commit id is also stored as the zip comment.
"""
import datetime as dt
import json
import subprocess
import sys
import zipfile

from src import config

PATHS = ["run_all.py", "src", "config", "tools", "tests", "requirements.txt"]


def git(*args):
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True)


def main():
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise SystemExit("git not available or not a git repository: " + head.stderr)
    commit = head.stdout.strip()
    dirty = git("status", "--porcelain", "--", *PATHS).stdout.strip()
    if dirty:
        raise SystemExit("Uncommitted changes in code/config (commit and push first):\n" + dirty)
    out = config.ROOT / "notebook_bundle.zip"
    # autocrlf off: the notebook is Linux, and shell scripts with Windows line endings do not run there
    r = git("-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=zip", "--prefix=valence-audit/",
            "-o", str(out), "HEAD", "--", *PATHS)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    upstream = git("rev-parse", "@{u}").stdout.strip()
    info = dict(commit=commit, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                pushed=(upstream == commit), paths=PATHS)
    with zipfile.ZipFile(out, "a") as z:
        z.writestr("valence-audit/BUNDLE_INFO.json", json.dumps(info, indent=1))
    print(f"Written: {out}  ({out.stat().st_size / 1e6:.1f} MB)  commit {commit[:10]}")
    if not info["pushed"]:
        print("Note: this commit is not on the remote yet. Run `git push` so that the code is public before collection.")


if __name__ == "__main__":
    sys.exit(main())
