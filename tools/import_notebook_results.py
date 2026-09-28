"""Bring the GPU-notebook results into the repository WITHOUT showing any response text.

    python -m tools.import_notebook_results r1_notebook_results_<UTC>.tar.gz

Checks before writing anything:
  - every file in the archive lies under data/r1/raw/, data/r1/budget12/raw/, data/r1/endpoints/,
    data/r1/logs/notebook_*/ or data/model_weights/;
  - no existing file would be overwritten with different content (a later archive may only add
    lines to files imported from an earlier one);
  - the notebook ran committed code: the commit in BUNDLE_INFO.json is in this repository's history
    and the code (src/, config/, tools/, run_all.py) is unchanged between that commit and HEAD.
Then extracts and prints the record counts (python run_all.py --r1-status).
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

from src import config

ALLOWED = ("data/r1/raw/", "data/r1/budget12/raw/", "data/r1/endpoints/", "data/r1/logs/notebook_",
           "data/model_weights/")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = pathlib.Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    root = config.ROOT
    with tarfile.open(src, "r:gz") as t:
        members = [m for m in t.getmembers() if m.isfile()]
        problems, plan, info = [], [], None
        for m in members:
            name = m.name.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/") or not name.startswith(ALLOWED):
                problems.append(f"unexpected path: {name}")
                continue
            data = t.extractfile(m).read()
            dest = root / name
            if dest.exists():
                old = dest.read_bytes()
                if _sha(old) != _sha(data) and not data.startswith(old):
                    # a later archive may only ADD lines to files imported from an earlier one
                    problems.append(f"would overwrite a different existing file: {name}")
                    continue
            if name.endswith("run_info.json"):
                try:
                    info = json.loads(data.decode("utf-8"))
                except Exception:
                    pass
            plan.append((dest, data))
    if problems:
        raise SystemExit("Nothing was imported:\n  " + "\n  ".join(problems))

    commit = ((info or {}).get("bundle") or {}).get("commit")
    if not commit:
        print("WARNING: the archive does not record the commit the notebook ran (BUNDLE_INFO.json missing).")
    else:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root)
        if anc.returncode != 0:
            raise SystemExit(f"The notebook ran commit {commit[:10]}, which is not in this repository's history. "
                             f"Nothing was imported.")
        diff = subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--", "src", "config", "tools", "run_all.py"],
                              cwd=root)
        if diff.returncode != 0:
            print(f"WARNING: code changed between the notebook's commit {commit[:10]} and HEAD "
                  f"(git diff {commit[:10]} HEAD -- src config tools run_all.py). Tell the assistant.")
        else:
            print(f"Notebook ran commit {commit[:10]}; code identical to HEAD.")

    new = grown = same = 0
    for dest, data in plan:
        if dest.exists():
            if dest.read_bytes() == data:
                same += 1
                continue
            grown += 1
        else:
            new += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"Imported: {new} new file(s), {grown} extended, {same} already present and identical.")
    from src import r1
    r1.status()


if __name__ == "__main__":
    main()
