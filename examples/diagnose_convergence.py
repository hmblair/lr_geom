"""
Diagnose convergence issues in the Structure VAE.

This script traces through the VAE layer by layer to find where
the signal dies or becomes too small.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path.home() / "software/ciffy"))

import lr_geom as lg
from lr_geom.vae import EquivariantVAE, kl_divergence
import ciffy


def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load real structures
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    data_dir = Path.home() / "data/pdb130"
    structures = []

    print(f"Looking in: {data_dir}")
    cif_files = sorted(data_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files")

    for cif_file in cif_files:
        try:
            polymer = ciffy.load(str(cif_file), backend="torch")
            n_atoms = polymer.size()
            if n_atoms > 700 or n_atoms < 20:
                continue
            polymer, _ = polymer.center()
            coords = polymer.coordinates.float()
            coord_scale = coords.std()
            coords_normalized = coords / coord_scale
            structures.append({
                "coords": coords_normalized.to(device),
                "coord_scale": coord_scale.item(),
                "atoms": polymer.atoms.long().to(device).clamp(min=0),
                "id": polymer.id(),
                "polymer": polymer.to(device),
            })
            print(f"  Loaded {cif_file.name}: {n_atoms} atoms")
            if len(structures) >= 1:
                break
        except Exception as e:
            print(f"  Error loading {cif_file.name}: {e}")
            continue

    if not structures:
        print("No structures loaded!")
        return

    s = structures[0]
    print(f"\nUsing: {s['id']} with {s['coords'].shape[0]} atoms")
    print(f"Coord scale: {s['coord_scale']:.2f}Å")
    print(f"Normalized coords range: [{s['coords'].min():.3f}, {s['coords'].max():.3f}]")
    print()

    # Create model
    print("=" * 60)
    print("MODEL SETUP")
    print("=" * 60)

    embed_dim = 8
    latent_mult = 4
    hidden_mult = 16

    in_repr = lg.Repr([0], mult=embed_dim)
    latent_repr = lg.Repr([0, 1], mult=latent_mult)
    out_repr = lg.Repr([1], mult=1)
    hidden_repr = lg.Repr([0, 1], mult=hidden_mult)

    print(f"in_repr: {in_repr.lvals} x {in_repr.mult} = dim {in_repr.dim()}")
    print(f"hidden_repr: {hidden_repr.lvals} x {hidden_repr.mult} = dim {hidden_repr.dim()}")
    print(f"latent_repr: {latent_repr.lvals} x {latent_repr.mult} = dim {latent_repr.dim()}")
    print(f"out_repr: {out_repr.lvals} x {out_repr.mult} = dim {out_repr.dim()}")
    print()

    embedding = nn.Embedding(ciffy.NUM_ATOMS, embed_dim).to(device)
    vae = EquivariantVAE(
        in_repr=in_repr,
        latent_repr=latent_repr,
        out_repr=out_repr,
        hidden_repr=hidden_repr,
        encoder_layers=2,
        decoder_layers=2,
        k_neighbors=16,
        nheads=4,
        dropout=0.0,
        residual_scale=1.0,
    ).to(device)

    num_params = sum(p.numel() for p in vae.parameters()) + sum(p.numel() for p in embedding.parameters())
    print(f"Total parameters: {num_params:,}")
    print()

    # Helper to print stats
    def print_stats(name, tensor):
        if tensor.numel() == 0:
            print(f"  {name}: EMPTY")
            return
        print(f"  {name}: shape={list(tensor.shape)}, "
              f"range=[{tensor.min():.4f}, {tensor.max():.4f}], "
              f"std={tensor.std():.4f}")

    # Register hooks to capture intermediate outputs
    print("=" * 60)
    print("LAYER-BY-LAYER TRACE (forward pass)")
    print("=" * 60)

    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks on key layers
    hooks = []

    # Encoder layers - with detailed attention tracing
    for i, layer in enumerate(vae.encoder.layers):
        hooks.append(layer.ln1.register_forward_hook(make_hook(f"enc.layer{i}.ln1")))
        # Detailed attention hooks
        hooks.append(layer.attn.proj_q.register_forward_hook(make_hook(f"enc.layer{i}.attn.proj_q")))
        hooks.append(layer.attn.conv_k.register_forward_hook(make_hook(f"enc.layer{i}.attn.conv_k")))
        hooks.append(layer.attn.conv_v.register_forward_hook(make_hook(f"enc.layer{i}.attn.conv_v")))
        hooks.append(layer.attn.out_proj.register_forward_hook(make_hook(f"enc.layer{i}.attn.out_proj")))
        hooks.append(layer.attn.register_forward_hook(make_hook(f"enc.layer{i}.attn")))
        if layer.transition is not None:
            hooks.append(layer.transition.register_forward_hook(make_hook(f"enc.layer{i}.transition")))

    hooks.append(vae.encoder.final_ln.register_forward_hook(make_hook("enc.final_ln")))
    hooks.append(vae.encoder.proj.register_forward_hook(make_hook("enc.proj")))

    # Variational head
    hooks.append(vae.var_head.mu_head.register_forward_hook(make_hook("var_head.mu")))
    hooks.append(vae.var_head.logvar_head.register_forward_hook(make_hook("var_head.logvar")))

    # Decoder layers
    for i, layer in enumerate(vae.decoder.layers):
        hooks.append(layer.ln1.register_forward_hook(make_hook(f"dec.layer{i}.ln1")))
        hooks.append(layer.attn.register_forward_hook(make_hook(f"dec.layer{i}.attn")))
        if layer.transition is not None:
            hooks.append(layer.transition.register_forward_hook(make_hook(f"dec.layer{i}.transition")))

    hooks.append(vae.decoder.final_ln.register_forward_hook(make_hook("dec.final_ln")))
    hooks.append(vae.decoder.proj.register_forward_hook(make_hook("dec.proj")))

    # Run forward pass
    coords = s["coords"]
    atoms = s["atoms"]

    with torch.no_grad():
        print("\n1. INPUT")
        features = embedding(atoms).unsqueeze(-1)
        print_stats("features (embedded)", features)

        print("\n2. VAE FORWARD")
        recon, mu, logvar = vae(coords, features)

        # Print all captured activations
        print("\n3. CAPTURED ACTIVATIONS")
        for name, act in activations.items():
            print_stats(name, act)

        print("\n4. FINAL OUTPUT")
        coords_pred = recon[:, 0, :]
        print_stats("reconstruction", recon)
        print_stats("coords_pred", coords_pred)
        print_stats("coords_target", coords)

        print(f"\nMSE: {((coords_pred - coords) ** 2).mean():.6f}")
        print(f"Target std: {coords.std():.4f}, Pred std: {coords_pred.std():.4f}")
        print(f"Ratio (pred/target std): {coords_pred.std() / coords.std():.4f}")

    # Clean up hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    diagnose()
