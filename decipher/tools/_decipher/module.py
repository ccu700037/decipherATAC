from typing import Sequence, List

import numpy as np
import torch
import torch.nn as nn


class ConditionalDenseNN(torch.nn.Module):
    """Dense neural network with multiple outputs, optionally conditioned on a context variable.

    (Derived from pyro.nn.dense_nn.ConditionalDenseNN with some modifications [1])

    Parameters
    ----------
    input_dim : int
        Dimension of the input
    hidden_dims : sequence of ints
        Dimensions of the hidden layers (excluding the output layer)
    output_dims : sequence of ints (optional)
        Dimensions of each output layer
        Default: (1,)
    context_dim : int (optional)
        Dimension of the context input.
        Default: 0. No context input.
    deep_context_injection : bool (optional)
        If True, inject the context into every hidden layer.
        If False, only inject the context into the first hidden layer (concatenated with the input).
        Default: False.
    activation : torch.nn.Module (optional)
        Activation function to use between hidden layers (not applied to the outputs).
        Default: torch.nn.ReLU()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dims: Sequence = (1,),
        context_dim: int = 0,
        deep_context_injection: bool = False,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.deep_context_injection = deep_context_injection
        self.n_output_layers = len(self.output_dims)
        self.output_total_dim = sum(self.output_dims)

        # The multiple outputs are computed as a single output layer, and then split
        indices = np.concatenate(([0], np.cumsum(self.output_dims)))
        self.output_slices = [slice(s, e) for s, e in zip(indices[:-1], indices[1:])]

        # Create masked layers
        deep_context_dim = self.context_dim if self.deep_context_injection else 0
        layers = []
        batch_norms = []
        if len(hidden_dims):
            layers.append(torch.nn.Linear(input_dim + context_dim, hidden_dims[0]))
            batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
            for i in range(1, len(hidden_dims)):
                layers.append(
                    torch.nn.Linear(hidden_dims[i - 1] + deep_context_dim, hidden_dims[i])
                )
                batch_norms.append(nn.BatchNorm1d(hidden_dims[i]))

            layers.append(
                torch.nn.Linear(hidden_dims[-1] + deep_context_dim, self.output_total_dim)
            )
        else:
            layers.append(torch.nn.Linear(input_dim + context_dim, self.output_total_dim))

        self.layers = torch.nn.ModuleList(layers)

        self.f = activation
        self.batch_norms = torch.nn.ModuleList(batch_norms)

    def forward(self, x, context=None):
        if context is not None:
            # We must be able to broadcast the size of the context over the input
            context = context.expand(x.size()[:-1] + (context.size(-1),))

        h = x
        for i, layer in enumerate(self.layers):
            if self.context_dim > 0 and (self.deep_context_injection or i == 0):
                h = torch.cat([context, h], dim=-1)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.batch_norms[i](h)
                h = self.f(h)

        if self.n_output_layers == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_total_dim])

            if self.n_output_layers == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.output_slices])

class ModularEncoder(nn.Module):
    def __init__(self, gene_modules, hidden_dim=32):
        super().__init__()
        self.gene_modules = gene_modules

        n_modules = len(gene_modules)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(genes) + n_modules, 64),  # was max(128, len(genes)*2)
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32),                       # was max(64, len(genes))
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
            for genes in gene_modules
        ])
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_init_weights)

    def forward(self, x):
        # Compute mean expression per module — fixed aggregation, no learned params
        module_means = torch.stack([
            x[:, gene_idx].mean(dim=-1) 
            for gene_idx in self.gene_modules
        ], dim=-1)  # (batch, n_modules)
        # Detach so this summary cannot be backpropped through to cheat
        module_means = module_means.detach()
        
        locs, scales = [], []
        for i, (encoder, gene_idx) in enumerate(zip(self.encoders, self.gene_modules)):
            x_mod = x[:, gene_idx]
            # Concatenate own genes + mean of ALL modules (including self)
            x_in = torch.cat([x_mod, module_means], dim=-1)
            out = encoder(x_in)
            locs.append(out[:, 0:1])
            scales.append(out[:, 1:2])
        z_loc = torch.cat(locs, dim=-1)
        z_scale = torch.cat(scales, dim=-1)
        return z_loc, z_scale