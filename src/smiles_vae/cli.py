"""Command line interface: ``python -m smiles_vae``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import describe, validity_rate
from .data import DEFAULT_SMILES, load_smiles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smiles-vae", description="Train a SMILES VAE and generate molecules."
    )
    p.add_argument("--data", type=Path, help="text file with one SMILES per line")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", type=Path, help="save the loss curve to this PNG")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imported here so `--help` stays fast and does not require TensorFlow.
    from .pipeline import generate, train

    smiles = load_smiles(args.data) if args.data else DEFAULT_SMILES
    print(f"Training on {len(smiles)} molecules for {args.epochs} epochs")
    t = train(
        smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    out = generate(t, args.num_samples, seed=args.seed)
    print("Generated:", out)
    print(f"Validity: {validity_rate(out):.0%}")
    for s in out:
        d = describe(s)
        if d is None:
            print(f"  invalid  {s!r}")
        else:
            print(
                f"  {s}  MW={d.mol_wt:.2f} LogP={d.log_p:.2f} "
                f"HBD={d.h_donors} HBA={d.h_acceptors}"
            )
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.plot(t.history["loss"], label="train")
        if "val_loss" in t.history:
            plt.plot(t.history["val_loss"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Saved loss curve to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
