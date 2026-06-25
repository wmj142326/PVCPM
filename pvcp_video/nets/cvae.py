#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.nn import Linear, Module, Sequential

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB


class CVAE(Module):
    def __init__(
        self,
        node_n=16,
        hidden_dim=256,
        z_dim=64,
        dct_n=10,
        dropout_rate=0,
    ):
        super().__init__()

        self.node_n = node_n
        self.dct_n = dct_n
        self.z_dim = z_dim
        enc_input_dim = 3 * dct_n * 2
        dec_input_dim = 3 * dct_n + z_dim

        self.enc = Sequential(
            GraphConvBlock(
                in_len=enc_input_dim,
                out_len=hidden_dim,
                in_node_n=node_n,
                out_node_n=node_n,
                dropout_rate=dropout_rate,
                bias=True,
                residual=False,
            ),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
        )

        self.mean = Linear(hidden_dim * node_n, z_dim)
        self.logvar = Linear(hidden_dim * node_n, z_dim)

        self.dec = Sequential(
            GraphConvBlock(
                in_len=dec_input_dim,
                out_len=hidden_dim,
                in_node_n=node_n,
                out_node_n=node_n,
                dropout_rate=dropout_rate,
                bias=True,
                residual=False,
            ),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            GraphConv(hidden_dim, 3 * dct_n, in_node_n=node_n, out_node_n=node_n, bias=True),
        )

    def _sample(self, mean, logvar):
        return torch.exp(0.5 * logvar) * torch.randn_like(logvar) + mean

    def forward(self, condition, data):
        b, _, _ = condition.shape
        posterior_input = torch.cat((condition, data), dim=-1)

        feature = self.enc(posterior_input)
        mean = self.mean(feature.view(b, -1))
        logvar = self.logvar(feature.view(b, -1))
        z = self._sample(mean, logvar)

        decoder_input = torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1)
        out = self.dec(decoder_input)
        out = out + condition
        return out, mean, logvar

    def inference(self, condition, z):
        decoder_input = torch.cat((condition, z.unsqueeze(dim=1).repeat([1, self.node_n, 1])), dim=-1)
        out = self.dec(decoder_input)
        out = out + condition
        return out
