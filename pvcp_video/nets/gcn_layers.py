#!/usr/bin/env python
# encoding: utf-8

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConv(nn.Module):
    def __init__(self, in_len, out_len, in_node_n=66, out_node_n=66, bias=True):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_node_n = in_node_n
        self.out_node_n = out_node_n
        self.weight = Parameter(torch.FloatTensor(in_len, out_len))
        self.att = Parameter(torch.FloatTensor(in_node_n, out_node_n))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_len))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        features = torch.matmul(input, self.weight)
        output = torch.matmul(features.permute(0, 2, 1).contiguous(), self.att).permute(0, 2, 1).contiguous()
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_len)
            + " -> "
            + str(self.out_len)
            + ") ("
            + str(self.in_node_n)
            + " -> "
            + str(self.out_node_n)
            + ")"
        )


class GraphConvBlock(nn.Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual
        self.out_len = out_len

        self.gcn = GraphConv(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias)
        self.bn = nn.BatchNorm1d(out_node_n * out_len)
        self.act = nn.Tanh()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = self.gcn(input)
        b, vc, t = x.shape
        x = self.bn(x.view(b, -1)).view(b, vc, t)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)

        if self.resual:
            return x + input
        return x


class ResGCB(nn.Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super().__init__()
        self.resual = residual
        self.gcb1 = GraphConvBlock(
            in_len,
            in_len,
            in_node_n=in_node_n,
            out_node_n=in_node_n,
            dropout_rate=dropout_rate,
            bias=bias,
            residual=False,
        )
        self.gcb2 = GraphConvBlock(
            in_len,
            out_len,
            in_node_n=in_node_n,
            out_node_n=out_node_n,
            dropout_rate=dropout_rate,
            bias=bias,
            residual=False,
        )

    def forward(self, input):
        x = self.gcb1(input)
        x = self.gcb2(x)

        if self.resual:
            return x + input
        return x
