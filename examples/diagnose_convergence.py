"""
Diagnose convergence issues in the Structure VAE.

This script runs detailed diagnostics to understand why the model
isn't learning to reconstruct coordinates.
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

    # Sanity check: verify l=1 spherical harmonics align with coordinates
    print("=" * 60)
    print("SPHERICAL HARMONICS ALIGNMENT CHECK")
    print("=" * 60)
    from lr_geom.equivariant import SphericalHarmonic
    sh = SphericalHarmonic(lmax=1)
    test_coords = torch.tensor([
        [1.0, 0.0, 0.0],  # x-axis
        [0.0, 1.0, 0.0],  # y-axis
        [0.0, 0.0, 1.0],  # z-axis
    ], device=device)
    sh_out = sh.to(device)(test_coords)
    l1_out = sh_out[:, 1:]  # Skip l=0 (first component)
    print(f"Input: x-axis [1,0,0] → l=1 output: [{l1_out[0,0]:.3f}, {l1_out[0,1]:.3f}, {l1_out[0,2]:.3f}]")
    print(f"Input: y-axis [0,1,0] → l=1 output: [{l1_out[1,0]:.3f}, {l1_out[1,1]:.3f}, {l1_out[1,2]:.3f}]")
    print(f"Input: z-axis [0,0,1] → l=1 output: [{l1_out[2,0]:.3f}, {l1_out[2,1]:.3f}, {l1_out[2,2]:.3f}]")
    print(f"\nExpected for correct (x,y,z) alignment:")
    print(f"  x-axis → [c, 0, 0]")
    print(f"  y-axis → [0, c, 0]")
    print(f"  z-axis → [0, 0, c]")
    print(f"where c is some constant (normalization factor)")
    print()

    # Test: Can a simple transformer learn to output coordinates directly?
    print("=" * 60)
    print("SIMPLE TRANSFORMER TEST (no VAE)")
    print("=" * 60)
    print("Testing if EquivariantTransformer can learn coords → coords mapping")
    print()

    # Create random point cloud
    N = 100
    coords_test = torch.randn(N, 3, device=device)
    coords_test = coords_test / coords_test.std()  # Normalize

    # Input: l=1 (the coordinates themselves)
    # Output: l=1 (should reconstruct coordinates)
    in_repr = lg.Repr([1], mult=1)
    out_repr = lg.Repr([1], mult=1)
    hidden_repr = lg.Repr([0, 1], mult=8)

    simple_model = lg.EquivariantTransformer(
        in_repr=in_repr,
        out_repr=out_repr,
        hidden_repr=hidden_repr,
        hidden_layers=2,
        k_neighbors=16,
        edge_dim=16,
        edge_hidden_dim=32,
        nheads=1,  # Must divide hidden_size (which is 3 for l=1)
        dropout=0.0,
    ).to(device)

    # Input features: just the coordinates as l=1 features
    features_test = coords_test.unsqueeze(1)  # (N, 1, 3)

    simple_opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    print("Training simple transformer to reconstruct coordinates...")
    simple_model.train()
    for i in range(100):
        simple_opt.zero_grad()
        out = simple_model(coords_test, features_test)
        pred = out[:, 0, :]  # (N, 3)
        loss = ((pred - coords_test) ** 2).mean()
        loss.backward()
        simple_opt.step()
        if i % 20 == 0:
            rmsd = ((pred - coords_test) ** 2).sum(dim=-1).mean().sqrt()
            print(f"  Iter {i:3d}: MSE={loss.item():.6f}, RMSD={rmsd.item():.4f}")

    simple_model.eval()
    with torch.no_grad():
        out = simple_model(coords_test, features_test)
        pred = out[:, 0, :]
        final_rmsd = ((pred - coords_test) ** 2).sum(dim=-1).mean().sqrt()
        print(f"\nFinal RMSD: {final_rmsd.item():.4f}")
        print(f"Sample predictions vs targets:")
        for i in range(3):
            print(f"  pred=[{pred[i,0]:.3f}, {pred[i,1]:.3f}, {pred[i,2]:.3f}] "
                  f"true=[{coords_test[i,0]:.3f}, {coords_test[i,1]:.3f}, {coords_test[i,2]:.3f}]")

    if final_rmsd > 0.5:  # 0.5 is reasonable for 100 iterations on unit-variance data
        print(f"\n⚠️  Simple transformer can't learn identity! RMSD={final_rmsd:.4f}")
        print("This suggests a fundamental issue with the architecture.")

        # Check basis elements
        print("\n--- CHECKING BASIS ELEMENTS ---")
        from lr_geom.equivariant import EquivariantBasis
        from lr_geom.representations import ProductRepr
        from copy import deepcopy

        # Create a ProductRepr like the transformer uses
        test_prepr = ProductRepr(deepcopy(in_repr), deepcopy(hidden_repr))
        test_basis = EquivariantBasis(test_prepr)

        # Compute basis for some displacements
        test_displacements = torch.randn(10, 3, device=device)
        b1, b2 = test_basis(test_displacements)

        print(f"ProductRepr: {in_repr.lvals} -> {hidden_repr.lvals}")
        print(f"Basis b1 shape: {b1.shape}, range: [{b1.min():.4f}, {b1.max():.4f}]")
        print(f"Basis b2 shape: {b2.shape}, range: [{b2.min():.4f}, {b2.max():.4f}]")

        if b1.abs().max() < 1e-6 or b2.abs().max() < 1e-6:
            print("⚠️  BASIS IS ZERO! This is the bug.")
        else:
            print("✓ Basis is non-zero")

        # Trace through the forward pass step by step
        print("\n--- TRACING FORWARD PASS ---")
        simple_model.eval()
        with torch.no_grad():
            # Get internal state by manually running forward pass
            N = coords_test.size(0)
            neighbor_idx = lg.build_knn_graph(coords_test, simple_model.k_neighbors)
            k = neighbor_idx.size(1)

            neighbor_coords = coords_test[neighbor_idx]
            displacements = coords_test.unsqueeze(1) - neighbor_coords

            distances = displacements.norm(dim=-1)
            edge_features = simple_model.rbf(distances)
            print(f"Edge features: shape={edge_features.shape}, range=[{edge_features.min():.4f}, {edge_features.max():.4f}]")

            # Compute bases
            all_bases = simple_model.bases(displacements.view(N * k, 3))
            basis_k = all_bases[0]
            basis_v = all_bases[1]
            b1_k, b2_k = basis_k
            b1_v, b2_v = basis_v
            print(f"Basis K: b1 range=[{b1_k.min():.4f}, {b1_k.max():.4f}], b2 range=[{b2_k.min():.4f}, {b2_k.max():.4f}]")
            print(f"Basis V: b1 range=[{b1_v.min():.4f}, {b1_v.max():.4f}], b2 range=[{b2_v.min():.4f}, {b2_v.max():.4f}]")

            # Check first layer components
            first_layer = simple_model.layers[0]
            print(f"\nFirst layer skip connection: {first_layer.skip}")

            # Check attention components
            attn = first_layer.attn
            print(f"Attention in_repr: {attn.repr.rep1.lvals}, mult={attn.repr.rep1.mult}")
            print(f"Attention out_repr: {attn.repr.rep2.lvals}, mult={attn.repr.rep2.mult}")

            # Test Q projection
            q_out = attn.proj_q(features_test)
            print(f"\nQ output: shape={q_out.shape}, range=[{q_out.min():.4f}, {q_out.max():.4f}]")

            if q_out.abs().max() < 1e-6:
                print("⚠️  Q IS ZERO!")

            # Reshape bases for layer call (N, k, ...)
            b1_k_r = b1_k.view(N, k, *b1_k.shape[1:])
            b2_k_r = b2_k.view(N, k, *b2_k.shape[1:])
            b1_v_r = b1_v.view(N, k, *b1_v.shape[1:])
            b2_v_r = b2_v.view(N, k, *b2_v.shape[1:])

            # Trace through block step by step
            print("\n--- TRACING INSIDE FIRST LAYER ---")

            # Step 1: LayerNorm
            lnorm = first_layer.ln1.lnorm
            if lnorm is None:
                print(f"\n✓ LayerNorm for mult=1: Using simple norm-based scaling (no LayerNorm)")
            else:
                print(f"\nLayerNorm normalized_shape: {lnorm.normalized_shape}")

            # Show RepNorm output
            norms = first_layer.ln1.norm(features_test)
            print(f"  RepNorm output: shape={norms.shape}, range=[{norms.min():.4f}, {norms.max():.4f}]")

            # Show why LayerNorm(1) fails
            test_ln = torch.nn.LayerNorm(1)
            test_input = torch.tensor([[1.0], [2.0], [3.0]])
            test_output = test_ln(test_input)
            print(f"  Demo: LayerNorm(1) on [1,2,3] -> {test_output.squeeze().tolist()}")

            ln1_out = first_layer.ln1(features_test)
            print(f"After ln1: shape={ln1_out.shape}, range=[{ln1_out.min():.4f}, {ln1_out.max():.4f}]")

            # Step 2: Inside attention - conv_k and conv_v
            src_idx = neighbor_idx.flatten()
            edge_feats_flat = edge_features.view(N * k, -1)
            b1_k_flat = b1_k_r.view(N * k, *b1_k_r.shape[2:])
            b2_k_flat = b2_k_r.view(N * k, *b2_k_r.shape[2:])
            b1_v_flat = b1_v_r.view(N * k, *b1_v_r.shape[2:])
            b2_v_flat = b2_v_r.view(N * k, *b2_v_r.shape[2:])

            # Keys
            keys = attn.conv_k((b1_k_flat, b2_k_flat), edge_feats_flat, ln1_out, src_idx)
            print(f"Keys (raw): shape={keys.shape}, range=[{keys.min():.4f}, {keys.max():.4f}]")

            # Values
            values = attn.conv_v((b1_v_flat, b2_v_flat), edge_feats_flat, ln1_out, src_idx)
            print(f"Values (raw): shape={values.shape}, range=[{values.min():.4f}, {values.max():.4f}]")

            if keys.abs().max() < 1e-6:
                print("⚠️  KEYS ARE ZERO!")
            if values.abs().max() < 1e-6:
                print("⚠️  VALUES ARE ZERO!")

            # Full layer
            layer_out = first_layer(
                (b1_k_r, b2_k_r), (b1_v_r, b2_v_r),
                features_test, edge_features, neighbor_idx, None
            )
            print(f"\nFirst layer output: shape={layer_out.shape}, range=[{layer_out.min():.4f}, {layer_out.max():.4f}]")

        # Debug: check gradients
        print("\n--- DEBUGGING ---")
        print(f"Input features shape: {features_test.shape}")
        print(f"Input features range: [{features_test.min():.3f}, {features_test.max():.3f}]")
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

        print(f"\nModel structure:")
        print(f"  in_repr: lvals={in_repr.lvals}, mult={in_repr.mult}, dim={in_repr.dim()}")
        print(f"  hidden_repr: lvals={hidden_repr.lvals}, mult={hidden_repr.mult}, dim={hidden_repr.dim()}")
        print(f"  out_repr: lvals={out_repr.lvals}, mult={out_repr.mult}, dim={out_repr.dim()}")

        # Check gradients
        print("\n--- Checking gradients ---")
        simple_model.train()
        simple_opt.zero_grad()
        out = simple_model(coords_test, features_test)
        pred = out[:, 0, :]
        loss = ((pred - coords_test) ** 2).mean()
        loss.backward()

        print("Gradient norms per layer:")
        for name, param in simple_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-8:
                    print(f"  {name}: grad={grad_norm:.2e} ← ZERO!")
                elif grad_norm > 1e-4:
                    print(f"  {name}: grad={grad_norm:.4f}")

    else:
        print(f"\n✓ Simple transformer CAN learn identity. Issue is likely in VAE.")
    print()

    # Load a single structure
    cif_dir = Path.home() / "data/pdb130"
    cif_files = sorted(cif_dir.glob("*.cif"))[:50]

    structures = []
    for cif_file in cif_files:
        try:
            polymer = ciffy.load(str(cif_file), backend="torch")
            if polymer.size() > 700:  # Atom count
                continue
            polymer, _ = polymer.center()
            coords = polymer.coordinates.float()
            # Normalize coordinates to unit variance for better optimization
            coord_scale = coords.std()
            coords_normalized = coords / coord_scale
            structures.append({
                "coords": coords_normalized.to(device),
                "coords_original": coords.to(device),
                "coord_scale": coord_scale.item(),
                "atoms": polymer.atoms.long().to(device).clamp(min=0),
                "id": polymer.id(),
                "polymer": polymer,  # Keep for ciffy.rmsd
            })
            if len(structures) >= 2:
                break
        except Exception as e:
            continue

    if not structures:
        print("No structures loaded!")
        return

    print(f"Loaded {len(structures)} structures")
    for s in structures:
        print(f"  {s['id']}: {s['coords'].shape[0]} atoms, scale={s['coord_scale']:.2f}Å")
    print()

    # Create a small model
    embed_dim = 8
    latent_mult = 4
    hidden_mult = 16

    class SmallVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(ciffy.NUM_ATOMS, embed_dim)

            self.in_repr = lg.Repr([0], mult=embed_dim)
            self.latent_repr = lg.Repr([0, 1], mult=latent_mult)
            self.out_repr = lg.Repr([1], mult=1)
            self.hidden_repr = lg.Repr([0, 1], mult=hidden_mult)

            self.vae = EquivariantVAE(
                in_repr=self.in_repr,
                latent_repr=self.latent_repr,
                out_repr=self.out_repr,
                hidden_repr=self.hidden_repr,
                encoder_layers=2,
                decoder_layers=2,
                k_neighbors=16,
                nheads=4,
                dropout=0.0,
                residual_scale=1.0,
            )

        def forward(self, coords, atoms):
            features = self.embedding(atoms).unsqueeze(-1)  # (N, embed_dim, 1)
            return self.vae(coords, features)

    model = SmallVAE().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Check initial outputs
    print("=" * 60)
    print("INITIAL STATE DIAGNOSTICS")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        s = structures[0]
        recon, mu, logvar = model(s["coords"], s["atoms"])
        coords_pred = recon[:, 0, :]  # (N, 3)

        print(f"\nInput coords stats (normalized, scale={s['coord_scale']:.2f}Å):")
        print(f"  Shape: {s['coords'].shape}")
        print(f"  Mean: {s['coords'].mean(dim=0).tolist()}")
        print(f"  Std: {s['coords'].std(dim=0).tolist()}")
        print(f"  Range: [{s['coords'].min():.3f}, {s['coords'].max():.3f}]")

        print(f"\nOutput coords stats:")
        print(f"  Shape: {coords_pred.shape}")
        print(f"  Mean: {coords_pred.mean(dim=0).tolist()}")
        print(f"  Std: {coords_pred.std(dim=0).tolist()}")
        print(f"  Range: [{coords_pred.min():.3f}, {coords_pred.max():.3f}]")

        print(f"\nLatent stats:")
        print(f"  mu shape: {mu.shape}, range: [{mu.min():.3f}, {mu.max():.3f}]")
        print(f"  logvar shape: {logvar.shape}, range: [{logvar.min():.3f}, {logvar.max():.3f}]")

        mse = ((coords_pred - s["coords"]) ** 2).mean()
        print(f"\nInitial MSE: {mse.item():.4f}")

    # Training
    print()
    print("=" * 60)
    print("TRAINING DIAGNOSTICS")
    print("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    losses = []
    for iteration in range(200):
        total_loss = 0.0

        for s in structures:
            optimizer.zero_grad()

            recon, mu, logvar = model(s["coords"], s["atoms"])
            coords_pred = recon[:, 0, :]

            recon_loss = ((coords_pred - s["coords"]) ** 2).mean()
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + 0.001 * kl_loss  # Very small KL weight

            loss.backward()

            # Check gradients
            if iteration == 0:
                print(f"\nGradient norms at iteration 0:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        param_norm = param.norm().item()
                        print(f"  {name}: grad={grad_norm:.6f}, param={param_norm:.6f}")

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(structures)
        losses.append(avg_loss)

        if iteration % 20 == 0 or iteration < 5:
            print(f"Iter {iteration:3d}: loss={avg_loss:.6f}")

    # Final check
    print()
    print("=" * 60)
    print("FINAL STATE DIAGNOSTICS")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        s = structures[0]
        recon, mu, logvar = model(s["coords"], s["atoms"])
        coords_pred = recon[:, 0, :]

        print(f"\nOutput coords stats after training (normalized):")
        print(f"  Mean: {coords_pred.mean(dim=0).tolist()}")
        print(f"  Std: {coords_pred.std(dim=0).tolist()}")
        print(f"  Range: [{coords_pred.min():.3f}, {coords_pred.max():.3f}]")

        mse = ((coords_pred - s["coords"]) ** 2).mean()

        # Compute Kabsch-aligned RMSD using ciffy (proper structural comparison)
        # Denormalize predictions back to original scale and move to CPU for ciffy
        coords_pred_angstrom = (coords_pred * s["coord_scale"]).cpu()
        pred_polymer = s["polymer"].with_coordinates(coords_pred_angstrom)
        kabsch_rmsd_sq = ciffy.rmsd(s["polymer"], pred_polymer, ciffy.MOLECULE)
        kabsch_rmsd = kabsch_rmsd_sq.sqrt().item()

        # Also compute raw RMSD (no alignment) for comparison
        rmsd_normalized = (((coords_pred - s["coords"]) ** 2).sum(dim=-1).mean() ** 0.5)
        rmsd_angstrom = rmsd_normalized.item() * s["coord_scale"]

        print(f"\nFinal MSE (normalized): {mse.item():.4f}")
        print(f"Final RMSD raw (no alignment): {rmsd_angstrom:.4f} Å")
        print(f"Final RMSD Kabsch-aligned: {kabsch_rmsd:.4f} Å")

        # Sample comparison (in normalized scale)
        print(f"\nSample predictions vs targets (first 5 atoms, normalized):")
        for i in range(min(5, coords_pred.shape[0])):
            pred = coords_pred[i].tolist()
            true = s["coords"][i].tolist()
            print(f"  Atom {i}: pred=[{pred[0]:7.3f}, {pred[1]:7.3f}, {pred[2]:7.3f}] "
                  f"true=[{true[0]:7.3f}, {true[1]:7.3f}, {true[2]:7.3f}]")

    # Loss curve
    print(f"\nLoss progression:")
    print(f"  Start: {losses[0]:.6f}")
    print(f"  End: {losses[-1]:.6f}")
    print(f"  Ratio: {losses[-1]/losses[0]:.4f}")

    if losses[-1] > losses[0] * 0.9:
        print("\n⚠️  WARNING: Loss barely decreased! Model is not learning.")
    elif losses[-1] > losses[0] * 0.5:
        print("\n⚠️  WARNING: Loss decreased slowly. Possible optimization issue.")
    else:
        print("\n✓ Loss decreased significantly.")


if __name__ == "__main__":
    diagnose()
