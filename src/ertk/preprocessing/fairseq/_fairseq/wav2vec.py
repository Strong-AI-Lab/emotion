# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
    TransposeLast,
)


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        base_wav2vec_architecture(args)

        self.prediction_steps = args.prediction_steps
        offset = args.offset

        if args.activation == "relu":
            activation = nn.ReLU()
        elif args.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + args.activation)

        if args.encoder == "cnn":
            feature_enc_layers = eval(args.conv_feature_layers)
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                log_compression=args.log_compression,
                skip_connections=args.skip_connections_feat,
                residual_scale=args.residual_scale,
                non_affine_group_norm=args.non_affine_group_norm,
                activation=activation,
            )
            embed = feature_enc_layers[-1][0]
        else:
            raise Exception("unknown encoder type " + args.encoder)

        self.vector_quantizer = None
        if args.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=embed,
                num_vars=args.vq_vars,
                temp=eval(args.vq_temp),
                groups=args.vq_groups,
                combine_groups=args.combine_groups,
                vq_dim=args.vq_dim if args.vq_dim > 0 else embed,
                time_first=False,
                activation=activation,
                weight_proj_depth=args.vq_depth,
                weight_proj_factor=2,
            )
        elif args.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=embed,
                num_vars=args.vq_vars,
                groups=args.vq_groups,
                combine_groups=args.combine_groups,
                vq_dim=args.vq_dim if args.vq_dim > 0 else embed,
                time_first=False,
                gamma=args.vq_gamma,
            )
        else:
            assert (
                args.vq_type == "none" or args.vq_type is None
            ), "Unknown quantizer type"

        if args.offset == "auto":
            assert args.encoder == "cnn"
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)

        offset = int(offset)

        def make_aggregator():
            if args.aggregator == "cnn":
                agg_layers = eval(args.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                feature_aggregator = ConvAggegator(
                    conv_layers=agg_layers,
                    embed=embed,
                    dropout=args.dropout,
                    skip_connections=args.skip_connections_agg,
                    residual_scale=args.residual_scale,
                    non_affine_group_norm=args.non_affine_group_norm,
                    conv_bias=not args.no_conv_bias,
                    zero_pad=args.agg_zero_pad,
                    activation=activation,
                )
            elif args.aggregator == "gru":
                agg_dim = args.gru_dim
                feature_aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=embed,
                        hidden_size=agg_dim,
                        num_layers=1,
                        dropout=args.dropout,
                    ),
                    TransposeLast(deconstruct_idx=0),
                )
            else:
                raise Exception("unknown aggregator type " + args.aggregator)

            return feature_aggregator, agg_dim

        self.feature_aggregator, agg_dim = make_aggregator()

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=args.prediction_steps,
            n_negatives=args.num_negatives,
            cross_sample_negatives=args.cross_sample_negatives,
            sample_distance=args.sample_distance,
            dropout=args.dropout,
            offset=offset,
            balanced_classes=args.balanced_classes,
            infonce=args.infonce,
        )

        self.dropout_feats = nn.Dropout(p=args.dropout_features)
        self.dropout_agg = nn.Dropout(p=args.dropout_agg)

        if args.project_features == "none":
            self.project_features = None
        elif args.project_features == "same":
            self.project_features = self.feature_aggregator
        elif args.project_features == "new":
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)

        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers,
        dropout,
        log_compression,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class ConvAggegator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        conv_bias,
        zero_pad,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class Wav2VecPredictionsModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        prediction_steps,
        n_negatives,
        cross_sample_negatives,
        sample_distance,
        dropout,
        offset,
        balanced_classes,
        infonce,
    ):
        super().__init__()

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)

        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)

        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )
        if self.infonce:
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            labels = torch.zeros_like(predictions)
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                labels[start : start + pos_num] = 1.0
                if weights is not None:
                    weights[start : start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        if self.infonce:
            predictions = predictions.view(-1, copies)
        else:
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels


def base_wav2vec_architecture(args):
    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.conv_aggregator_layers = getattr(
        args, "conv_aggregator_layers", "[(512, 3, 1)] * 9"
    )

    args.prediction_steps = getattr(args, "prediction_steps", 12)
    args.num_negatives = getattr(args, "num_negatives", 1)
    args.sample_distance = getattr(args, "sample_distance", None)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)

    args.dropout = getattr(args, "dropout", 0.0)
    args.dropout_features = getattr(args, "dropout_features", 0.0)
    args.dropout_agg = getattr(args, "dropout_agg", 0.0)
    args.encoder = getattr(args, "encoder", "cnn")
    args.aggregator = getattr(args, "aggregator", "cnn")

    args.skip_connections_feat = getattr(args, "skip_connections_feat", False)
    args.skip_connections_agg = getattr(args, "skip_connections_agg", False)
    args.residual_scale = getattr(args, "residual_scale", 0.5)

    args.gru_dim = getattr(args, "gru_dim", 512)

    args.no_conv_bias = getattr(args, "no_conv_bias", False)
    args.agg_zero_pad = getattr(args, "agg_zero_pad", False)

    args.log_compression = getattr(args, "log_compression", False)

    args.balanced_classes = getattr(args, "balanced_classes", False)
    args.infonce = getattr(args, "infonce", False)
    args.project_features = getattr(args, "project_features", "none")

    args.non_affine_group_norm = getattr(args, "non_affine_group_norm", False)

    args.offset = getattr(args, "offset", "auto")

    args.activation = getattr(args, "activation", "relu")

    args.vq_type = getattr(args, "vq_type", "none")
    args.vq_vars = getattr(args, "vq_vars", 320)
    args.vq_groups = getattr(args, "vq_groups", 2)
    args.vq_dim = getattr(args, "vq_dim", 0)
    args.vq_depth = getattr(args, "vq_depth", 1)
    args.combine_groups = getattr(args, "combine_groups", False)
    args.vq_temp = getattr(args, "vq_temp", "(2.0, 0.5, 0.999995)")
    args.vq_gamma = getattr(args, "vq_gamma", 0.25)
