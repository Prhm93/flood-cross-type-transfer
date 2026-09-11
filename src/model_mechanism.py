"""
model_mechanism.py
===================

A model that is TOLD which flood mechanism it is looking at: a flag
saying "this is a point-source flood" or "this is a distributed
(rainfall-like) flood".

The tag goes in as two extra STATIC channels (one-hot: is_point,
is_distributed), because the mechanism does not change over time inside
one simulation. Everything else is identical to FloodGNN.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.model import MessagePassingLayer


class FloodGNNMech(nn.Module):
    """
    Same as FloodGNN, except the static input carries elevation PLUS a
    2-value mechanism tag, so static_dim is 3 instead of 1.
    """

    def __init__(self, variant='vector', hidden=64, num_layers=3):
        super().__init__()
        self.variant = variant
        # +2 input channels for the tag
        in_channels = 5 + 2 if variant == 'vector' else 4 + 2

        self.encoder = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU())
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden, hidden) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, dynamic, static_with_tag, edge_index):
        """
        dynamic         : (N, 4)  depth, vx, vy, rain
        static_with_tag : (N, 3)  elevation, is_point, is_distributed
        """
        if self.variant == 'vector':
            x = torch.cat([dynamic, static_with_tag], dim=-1)
        else:
            # scalar variant: collapse the two velocity components to a speed
            depth = dynamic[:, :1]
            vx, vy = dynamic[:, 1:2], dynamic[:, 2:3]
            speed = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
            rain = dynamic[:, 3:]
            x = torch.cat([depth, speed, rain, static_with_tag], dim=-1)

        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.decoder(h)


def add_mechanism_tag(samples, mechanism):
    """
    Return copies of the samples with a 2-column one-hot tag stuck onto
    nodes_static: [elevation, is_point, is_distributed].

    mechanism: 'point' or 'distributed'. 'breach' and 'harvey' map the
    same way, since breach is a point source and Harvey is distributed.
    """
    import numpy as np
    is_point = 1.0 if mechanism in ('point', 'breach') else 0.0
    is_dist = 1.0 if mechanism in ('distributed', 'harvey') else 0.0

    out = []
    for s in samples:
        s2 = dict(s)
        elev = s['nodes_static']  # (N, 1)
        n = elev.shape[0]
        tag = np.tile(np.array([[is_point, is_dist]], dtype=np.float32), (n, 1))
        s2['nodes_static'] = np.concatenate([elev, tag], axis=1).astype(np.float32)
        out.append(s2)
    return out


if __name__ == '__main__':
    # small self-test: does it build, run forward, and backprop
    torch.manual_seed(0)
    m = FloodGNNMech('vector', hidden=16, num_layers=2)
    dyn = torch.randn(10, 4)
    sta = torch.randn(10, 3)   # elevation + 2-value tag
    ei = torch.randint(0, 10, (2, 30))
    out = m(dyn, sta, ei)
    assert out.shape == (10, 3), out.shape
    out.sum().backward()

    import numpy as np
    fake = [{'nodes_static': np.random.rand(5, 1).astype(np.float32)}]
    tagged = add_mechanism_tag(fake, 'point')
    assert tagged[0]['nodes_static'].shape == (5, 3)
    assert (tagged[0]['nodes_static'][:, 1] == 1.0).all()
    assert (tagged[0]['nodes_static'][:, 2] == 0.0).all()
    print("model_mechanism.py self-test: OK")
