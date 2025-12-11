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
sys.path.insert(0, str(Path.home() / "academic/software/ciffy"))

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

    data_dir = Path.home() / "academic/data/rna-libraries/pdb/structures/pdb130"
    structures = []

    for cif_file in sorted(data_dir.glob("*.cif"))[:5]:
        try:
            polymer = ciffy.load(str(cif_file), backend="torch")
            if polymer.size() > 500 or polymer.size() < 50:
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
            if len(structures) >= 1:
                break
        except Exception as e:
            continue

    if not structures:
        print("No structures loaded!")
        return

    s = structures[0]
    print(f"Loaded: {s['id']} with {s['coords'].shape[0]} atoms")
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

    # Detailed layer-by-layer trace
    print("=" * 60)
    print("LAYER-BY-LAYER TRACE (forward pass)")
    print("=" * 60)

    coords = s["coords"]
    atoms = s["atoms"]

    def print_stats(name, tensor):
        if tensor.numel() == 0:
            print(f"  {name}: EMPTY")
            return
        print(f"  {name}: shape={list(tensor.shape)}, "
              f"range=[{tensor.min():.4f}, {tensor.max():.4f}], "
              f"std={tensor.std():.4f}, mean={tensor.abs().mean():.4f}")

    with torch.no_grad():
        # 1. Input embedding
        print("\n1. INPUT EMBEDDING")
        features = embedding(atoms).unsqueeze(-1)  # (N, embed_dim, 1)
        print_stats("atoms (input)", atoms.float())
        print_stats("features (embedded)", features)

        # 2. Encoder input projection
        print("\n2. ENCODER INPUT PROJECTION")
        enc_input = vae.encoder.input_proj(features)
        print_stats("encoder input_proj output", enc_input)

        # 3. Encoder layers
        print("\n3. ENCODER LAYERS")
        # We need to manually trace through encoder
        enc = vae.encoder
        neighbor_idx = lg.build_knn_graph(coords, enc.k_neighbors)
        displacements = coords[neighbor_idx] - coords.unsqueeze(1)
        edge_features = enc.rbf(displacements.norm(dim=-1))
        all_bases = enc.bases(displacements)

        x = enc_input
        for i, block in enumerate(enc.layers):
            # Get bases for this layer
            basis_k = all_bases[2 * i]
            basis_v = all_bases[2 * i + 1]

            # Attention
            x_ln = block.ln1(x)
            print_stats(f"  layer {i} after ln1", x_ln)

            attn_out = block.attn(x_ln, neighbor_idx, edge_features, basis_k, basis_v)
            print_stats(f"  layer {i} attn output", attn_out)

            if block.skip:
                x = attn_out + block.residual_scale * x
            else:
                x = attn_out
            print_stats(f"  layer {i} after residual", x)

            # Transition (if present)
            if block.transition is not None:
                x_ln2 = block.ln2(x)
                trans_out = block.transition(x_ln2)
                x = trans_out + block.residual_scale * x
                print_stats(f"  layer {i} after transition", x)

        encoder_output = enc.final_ln(x)
        print_stats("encoder final output", encoder_output)

        # 4. Variational head
        print("\n4. VARIATIONAL HEAD")
        mu = vae.var_head.mu_proj(encoder_output)
        print_stats("mu", mu)

        norms = lg.RepNorm(vae.var_head.in_repr)(encoder_output)
        print_stats("norms for logvar", norms)
        logvar = vae.var_head.logvar_net(norms)
        print_stats("logvar", logvar)

        # 5. Reparameterization
        print("\n5. REPARAMETERIZATION")
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std.unsqueeze(-1) * eps
        print_stats("z (latent)", z)

        # 6. Decoder
        print("\n6. DECODER")
        dec = vae.decoder
        dec_input = dec.input_proj(z)
        print_stats("decoder input_proj output", dec_input)

        # Decoder uses same graph structure
        dec_neighbor_idx = lg.build_knn_graph(coords, dec.k_neighbors)
        dec_displacements = coords[dec_neighbor_idx] - coords.unsqueeze(1)
        dec_edge_features = dec.rbf(dec_displacements.norm(dim=-1))
        dec_all_bases = dec.bases(dec_displacements)

        y = dec_input
        for i, block in enumerate(dec.layers):
            basis_k = dec_all_bases[2 * i]
            basis_v = dec_all_bases[2 * i + 1]

            y_ln = block.ln1(y)
            print_stats(f"  layer {i} after ln1", y_ln)

            attn_out = block.attn(y_ln, dec_neighbor_idx, dec_edge_features, basis_k, basis_v)
            print_stats(f"  layer {i} attn output", attn_out)

            if block.skip:
                y = attn_out + block.residual_scale * y
            else:
                y = attn_out
            print_stats(f"  layer {i} after residual", y)

            if block.transition is not None:
                y_ln2 = block.ln2(y)
                trans_out = block.transition(y_ln2)
                y = trans_out + block.residual_scale * y
                print_stats(f"  layer {i} after transition", y)

        decoder_output = dec.final_ln(y)
        print_stats("decoder final_ln output", decoder_output)

        # 7. Output projection
        print("\n7. OUTPUT PROJECTION")
        recon = dec.output_proj(decoder_output)
        print_stats("final reconstruction", recon)

        coords_pred = recon[:, 0, :]
        print_stats("coords_pred (extracted)", coords_pred)
        print_stats("coords_target", coords)

        print(f"\nMSE: {((coords_pred - coords) ** 2).mean():.6f}")
        print(f"Target std: {coords.std():.4f}, Pred std: {coords_pred.std():.4f}")
        print(f"Ratio (pred/target std): {coords_pred.std() / coords.std():.4f}")


if __name__ == "__main__":
    diagnose()
