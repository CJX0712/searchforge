"""SearchForge · 命令行入口。

星 (晨星) · 2026-10-07

用法：
  python -m searchforge.cli run    --seed 42
  python -m searchforge.cli bench  --seeds 42,43,44 --out benchmark.json
  python -m searchforge.cli tune   --trials 8
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .core.seed import set_all
from .pipeline.pipeline import SearchPipeline, benchmark, summarize


def _write(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_run(args) -> int:
    cfg = Config.from_env()
    cfg = type(cfg)(**{**cfg.as_dict(), "seed": args.seed})
    set_all(cfg.seed)
    pipe = SearchPipeline(cfg)
    res = pipe.run()
    print(
        summarize(
            {
                "summary": {
                    m: {k: {"mean": v, "std": 0.0, "values": [v]} for k, v in r.items()}
                    for m, r in res["methods"].items()
                },
                "seeds": [args.seed],
                "gate": {},
            }
        )
    )
    print(json.dumps(res["diagnosis"], ensure_ascii=False))
    if args.out:
        _write(args.out, res)
    return 0


def cmd_bench(args) -> int:
    cfg = Config.from_env()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = type(cfg)(**{**cfg.as_dict(), "seed": seeds[0], "n_seeds": len(seeds)})
    bench = benchmark(cfg, seeds)
    print(summarize(bench))
    if args.out:
        _write(args.out, bench)
        print(f"written -> {args.out}")
    return 0 if bench["gate"]["passed"] else 1


def cmd_tune(args) -> int:
    cfg = Config.from_env()
    from .hpo.tune import tune

    best, _study = tune(cfg, n_trials=args.trials)
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="searchforge", description="SearchForge CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", type=str, default="")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench")
    b.add_argument("--seeds", type=str, default="42,43,44")
    b.add_argument("--out", type=str, default="benchmark.json")
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("tune")
    t.add_argument("--trials", type=int, default=8)
    t.set_defaults(func=cmd_tune)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
