"""
model.py
========

A deliberately simple Graph Neural Network for flood prediction.

WHY SIMPLE
----------
The paper's contribution is NOT a clever architecture. It is the
TRANSFER TEST — training on one flood type and testing on another.
So the model needs to be:

    1. Simple enough that results are clearly about transfer, not
       about architecture tricks
    2. Standard enough that reviewers accept it as a fair test
    3. Small enough to train in minutes, not hours

WHAT IT DOES
------------
Given the current state of a flood (depth, velocity, rainfall at
each node) and static terrain, it predicts the NEXT state.

The prediction is a CHANGE (delta): how much depth, vx, vy will
change in one time step. The new state = old state + predicted delta.

This is called RESIDUAL prediction. It is standard practice —
SWE-GNN, FloodGNN-GRU, and most flood surrogates do the same thing.

ARCHITECTURE
------------
    Input per node: 4 dynamic channels + 1 static = 5 features
    
    3 message-passing layers:
        Each layer: every node collects messages from its neighbours,
        combines them with its own features, and updates itself.
        Think of it as: each cell asks its neighbours "what's your
        water like?" and uses the answers to update its own prediction.
    
    Output per node: 3 channels (delta_depth, delta_vx, delta_vy)
    
    Total parameters: ~50,000 (tiny by modern standards)

TWO VARIANTS (for the direction comparison)
-------------------------------------------
    'vector':  model receives vx and vy as separate channels (can see direction)
    'scalar':  model receives only speed = sqrt(vx² + vy²) (direction thrown away)

    Both predict the same 3 outputs. The question: does keeping direction
    help the model transfer better across flood types?
"""

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """
    One round of 'nodes talk to their neighbours'.

    Each node:
    1. Collects its neighbours' features
    2. Sums them up (aggregation)
    3. Combines the sum with its own features
    4. Passes through a small neural net
    5. Outputs updated features

    This is the simplest possible message-passing GNN layer.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # The 'thinking' network: takes [my features, neighbour sum] and outputs new features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        """
        x:          (N, D) node features
        edge_index: (2, E) which nodes connect — row 0 = source, row 1 = target
        """
        src, dst = edge_index  # each is (E,)

        # Gather source node features for each edge, then sum by destination
        messages = torch.zeros_like(x)
        messages.index_add_(0, dst, x[src])

        # Combine: [my own features, sum of neighbour features]
        combined = torch.cat([x, messages], dim=-1)  # (N, 2*D)

        # Think about it
        out = self.mlp(combined)
        out = self.norm(out + x if x.shape[-1] == out.shape[-1] else out)

        return out


class FloodGNN(nn.Module):
    """
    The complete model.

    variant='vector': input has vx and vy separately (5 input features)
    variant='scalar': input has speed only (4 input features)
    """

    def __init__(self, variant='vector', hidden=64, num_layers=3):
        super().__init__()
        self.variant = variant

        # Input size depends on variant
        if variant == 'vector':
            in_channels = 5  # depth + vx + vy + rain + elevation
        elif variant == 'scalar':
            in_channels = 4  # depth + speed + rain + elevation
        else:
            raise ValueError(f"variant must be 'vector' or 'scalar', got {variant}")

        # Encoder: raw features -> hidden dimension
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
        )

        # Message-passing layers
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden, hidden)
            for _ in range(num_layers)
        ])

        # Decoder: hidden -> predicted changes
        # Always predicts 3 channels: delta_depth, delta_vx, delta_vy
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, dynamic, static, edge_index):
        """
        dynamic:    (N, 4) current timestep — [depth, vx, vy, rain]
        static:     (N, 1) elevation
        edge_index: (2, E) graph connectivity

        Returns:    (N, 3) predicted change — [delta_depth, delta_vx, delta_vy]
        """
        if self.variant == 'vector':
            # Keep vx and vy as separate channels
            x = torch.cat([dynamic, static], dim=-1)  # (N, 5)
        else:
            # Replace vx, vy with scalar speed = sqrt(vx² + vy²)
            depth = dynamic[:, :1]
            vx = dynamic[:, 1:2]
            vy = dynamic[:, 2:3]
            speed = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)  # small number avoids sqrt(0)
            rain = dynamic[:, 3:]
            x = torch.cat([depth, speed, rain, static], dim=-1)  # (N, 4)

        # Encode
        h = self.encoder(x)

        # Message passing
        for layer in self.layers:
            h = layer(h, edge_index)

        # Decode to predicted changes
        delta = self.decoder(h)

        return delta


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Self-testing model.py ...\n")

    N = 16  # nodes
    E = 48  # edges

    # Fake data
    dynamic = torch.randn(N, 4)
    static = torch.randn(N, 1)
    edge_index = torch.randint(0, N, (2, E))

    # Test vector variant
    model_v = FloodGNN(variant='vector', hidden=32, num_layers=2)
    out_v = model_v(dynamic, static, edge_index)
    assert out_v.shape == (N, 3), f"vector output shape wrong: {out_v.shape}"
    print(f"  vector variant: {count_params(model_v):,} params, output {out_v.shape}  OK")

    # Test scalar variant
    model_s = FloodGNN(variant='scalar', hidden=32, num_layers=2)
    out_s = model_s(dynamic, static, edge_index)
    assert out_s.shape == (N, 3), f"scalar output shape wrong: {out_s.shape}"
    print(f"  scalar variant: {count_params(model_s):,} params, output {out_s.shape}  OK")

    # Check gradient flows
    loss = out_v.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model_v.parameters() if p.requires_grad)
    assert grad_ok, "some parameters got no gradient"
    print("  gradients flow to all parameters: OK")

    # Default size
    model_full = FloodGNN(variant='vector', hidden=64, num_layers=3)
    print(f"\n  Full model (hidden=64, 3 layers): {count_params(model_full):,} parameters")

    print("\nAll self-tests passed.")