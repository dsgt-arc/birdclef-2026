"""CLI entry point for exp 301.

    birdclef-301 lora    --config configs/lora-eat.yaml   --output-dir artifacts/checkpoints/eat-lora-r8
    birdclef-301 dapt    --config configs/dapt-eat.yaml   --output-dir artifacts/checkpoints/eat-dapt
    birdclef-301 ksshift --frozen <parquet> --adapted <parquet> --output artifacts/ks_eat.json

Extraction + probing of the adapted variants are NOT reimplemented here — they
delegate to `birdclef-300 extract --variant ... --adapter|--state-dict ...` and
`birdclef-300 probe` (see sbatch/reextract.sbatch and sbatch/probe.sbatch).
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="birdclef-301")
    sub = parser.add_subparsers(dest="command")

    p_lora = sub.add_parser("lora", help="Supervised-focal LoRA adaptation")
    p_lora.add_argument("--config", required=True)
    p_lora.add_argument("--output-dir", required=True)
    p_lora.add_argument("--max-steps", type=int, default=None, help="Cap steps (smoke test)")

    p_dapt = sub.add_parser("dapt", help="Soundscape DAPT — generic latent-bootstrap (baseline)")
    p_dapt.add_argument("--config", required=True)
    p_dapt.add_argument("--output-dir", required=True)
    p_dapt.add_argument("--max-steps", type=int, default=None, help="Cap steps (smoke test)")

    p_dn = sub.add_parser(
        "dapt-native",
        help="Soundscape DAPT — native objective (SSAMBA MPG/MPC | EAT data2vec)")
    p_dn.add_argument("--config", required=True)
    p_dn.add_argument("--output-dir", required=True)
    p_dn.add_argument("--max-steps", type=int, default=None, help="Cap steps (smoke test)")

    p_ks = sub.add_parser("ksshift", help="KS-shift between two embedding caches")
    p_ks.add_argument("--frozen", required=True, help="Parquet A (e.g. variant=frozen)")
    p_ks.add_argument("--adapted", required=True, help="Parquet B (e.g. variant=lora-r8)")
    p_ks.add_argument("--output", required=True)
    p_ks.add_argument("--label", default="frozen-vs-adapted")

    args = parser.parse_args(argv)

    if args.command == "lora":
        from birdclef_301.lora_adapt import run_lora

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_lora(cfg, args.output_dir, max_steps=args.max_steps)

    elif args.command == "dapt":
        from birdclef_301.dapt import run_dapt

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_dapt(cfg, args.output_dir, max_steps=args.max_steps)

    elif args.command == "dapt-native":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        encoder = cfg["encoder"]
        if encoder == "ssamba_hf":
            from birdclef_301.ssamba_dapt import run_ssamba_dapt

            run_ssamba_dapt(cfg, args.output_dir, max_steps=args.max_steps)
        elif encoder == "eat":
            from birdclef_301.eat_dapt import run_eat_dapt

            run_eat_dapt(cfg, args.output_dir, max_steps=args.max_steps)
        else:
            raise ValueError(
                f"dapt-native supports ssamba_hf|eat; got '{encoder}' "
                "(BEATs SSL needs its co-trained tokenizer — LoRA-only)")

    elif args.command == "ksshift":
        from birdclef_301.ks_shift import run_ksshift

        run_ksshift(args.frozen, args.adapted, args.output, label=args.label)

    else:
        parser.print_help()
        sys.exit(1)
