#!/usr/bin/env python3
"""Entry point.

  python run_all.py --mock          offline dry run of the full pipeline on fake data (writes data_mock/)
  python run_all.py --test          parser unit tests
  python run_all.py --reanalyze     measure -> analyze -> figures on the deposited June 2026 data (no API)
  python run_all.py --list-models   models your NVIDIA NIM key can call
  python run_all.py --smoke         1 model x 1 dimension live check
  python run_all.py --full          original June 2026 collection (resumable), then measure/analyze/figures
  python run_all.py --collect-r1    revision-round collection (config/collection_r1.yaml) -> data/r1/
        --group api                 only models served by an API (Bailian / ModelScope / NIM)
        --group local               only models served by the local vLLM server (GPU notebook)
        --models ID[,ID...]         only these models (June model ids)
        --allow-api-fallback        let notebook-first models use their API fallback (only if the notebook failed)
  python run_all.py --r1-status     records on disk per model and part (never shows response text)
  python run_all.py --analyze-r1    analyse the revision-round collection
Add --legacy-parser to --reanalyze to reproduce the submitted (v1.0.0) parsing.
"""
import sys
import asyncio
from src import config


def _post(mock=False):
    from src import measure, analyze, figures
    print("\n[measure]")
    measure.measure()
    print("\n[analyze]")
    analyze.run()
    print("\n[figures]")
    figures.run()


def _opt(name):
    """Value of `--name value` or `--name=value` on the command line, else None."""
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def main():
    a = set(sys.argv[1:])
    if "--legacy-parser" in a:
        config.EXP["measurement"]["rating_parser"] = "legacy"
    if "--test" in a:
        from tests import test_rating_parse as t
        for fn in (t.test_strict_cases, t.test_legacy_false_positives_documented,
                   t.test_strict_does_not_reproduce_false_positives):
            fn()
        print("parser tests passed")
    elif "--mock" in a:
        config.use_mock()
        from src import build_prompts, generate
        build_prompts.build()
        asyncio.run(generate.run(mock=True))
        _post(mock=True)
        from src import r1
        r1.collect(mock=True)
        r1.analyze(mock=True)
        print("\nMock run complete: data_mock/ (fake data; not evidence about any model).")
    elif "--reanalyze" in a:
        _post()
    elif "--list-models" in a:
        from src.providers import NimProvider
        ids = asyncio.run(NimProvider().list_models())
        print(f"\n{len(ids)} models available:")
        for i in sorted(ids):
            print("  -", i)
    elif "--smoke" in a or "--full" in a:
        from src import build_prompts, generate
        build_prompts.build()
        asyncio.run(generate.run(smoke="--smoke" in a))
        _post()
    elif "--collect-r1" in a:
        from src import r1
        only = _opt("--models")
        group = _opt("--group")
        if group not in (None, "api", "local"):
            raise SystemExit("--group must be 'api' or 'local'")
        r1.collect(only=[x.strip() for x in only.split(",") if x.strip()] if only else None, group=group,
                   allow_fallback="--allow-api-fallback" in a)
    elif "--r1-status" in a:
        from src import r1
        r1.status()
    elif "--analyze-r1" in a:
        from src import r1
        r1.analyze()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
