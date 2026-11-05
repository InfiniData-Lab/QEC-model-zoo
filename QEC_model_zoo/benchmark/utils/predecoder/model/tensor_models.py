import math
import torch
import torch.nn as nn


class MLPPreDecoder(nn.Module):
    """
    Global MLP predecoder.

    Input:
        x: [B, C_in, T, H, W]
    Output:
        y: [B, C_out, T, H, W]
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_dim=512,
        depth=3,
        dropout=0.1,
        debug_shapes=True,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.debug_shapes = debug_shapes

        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)

        layers = []
        dim = input_dim

        for _ in range(depth):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        b = x.shape[0]

        if self.debug_shapes and not hasattr(self, "_debug_printed"):
            print("[MLPPreDecoder] input:", tuple(x.shape))
            print("[MLPPreDecoder] expected input_shape:", self.input_shape)
            print("[MLPPreDecoder] output_shape:", self.output_shape)
            self._debug_printed = True

        y = self.net(x.reshape(b, -1))
        y = y.reshape(b, *self.output_shape)
        return y


class LSTMPreDecoder(nn.Module):
    """
    LSTM over syndrome rounds.

    For each spatial location (H, W), it treats T as a sequence.

    Input:
        x: [B, C_in, T, H, W]
    Output:
        y: [B, C_out, T, H, W]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        debug_shapes=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug_shapes = debug_shapes

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, out_channels),
        )

    def forward(self, x):
        b, c, t, h, w = x.shape

        if self.debug_shapes and not hasattr(self, "_debug_printed"):
            print("[LSTMPreDecoder] input:", tuple(x.shape))
            print("[LSTMPreDecoder] in_channels:", self.in_channels)
            print("[LSTMPreDecoder] out_channels:", self.out_channels)
            self._debug_printed = True

        # [B, C, T, H, W] -> [B, H, W, T, C]
        z = x.permute(0, 3, 4, 2, 1).contiguous()

        # [B*H*W, T, C]
        z = z.reshape(b * h * w, t, c)

        z, _ = self.lstm(z)

        # [B*H*W, T, C_out]
        y = self.proj(z)

        # [B, H, W, T, C_out] -> [B, C_out, T, H, W]
        y = y.reshape(b, h, w, t, self.out_channels)
        y = y.permute(0, 4, 3, 1, 2).contiguous()

        return y


# class TabMStylePreDecoder(nn.Module):
#     """
#     TabM-inspired tensor predecoder.

#     This is not the original TabM implementation. It is a tensor-compatible
#     ensemble-style MLP with multiple prediction heads.

#     Input:
#         x: [B, C_in, T, H, W]
#     Output:
#         y: [B, C_out, T, H, W]
#     """

#     def __init__(
#         self,
#         input_shape,
#         output_shape,
#         width=256,
#         depth=3,
#         num_heads=8,
#         dropout=0.1,
#         debug_shapes=True,
#     ):
#         super().__init__()
#         self.input_shape = tuple(input_shape)
#         self.output_shape = tuple(output_shape)
#         self.num_heads = num_heads
#         self.debug_shapes = debug_shapes

#         input_dim = math.prod(self.input_shape)
#         output_dim = math.prod(self.output_shape)

#         layers = [
#             nn.Linear(input_dim, width),
#             nn.LayerNorm(width),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         ]

#         for _ in range(max(depth - 1, 0)):
#             layers.extend([
#                 nn.Linear(width, width),
#                 nn.LayerNorm(width),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#             ])

#         self.backbone = nn.Sequential(*layers)
#         self.heads = nn.ModuleList([
#             nn.Linear(width, output_dim) for _ in range(num_heads)
#         ])

#     def forward(self, x):
#         b = x.shape[0]

#         if self.debug_shapes and not hasattr(self, "_debug_printed"):
#             print("[TabMStylePreDecoder] input:", tuple(x.shape))
#             print("[TabMStylePreDecoder] input_shape:", self.input_shape)
#             print("[TabMStylePreDecoder] output_shape:", self.output_shape)
#             print("[TabMStylePreDecoder] num_heads:", self.num_heads)
#             self._debug_printed = True

#         z = x.reshape(b, -1)
#         z = self.backbone(z)

#         # [B, K, output_dim]
#         ys = torch.stack([head(z) for head in self.heads], dim=1)

#         # Average logits across heads
#         y = ys.mean(dim=1)

#         return y.reshape(b, *self.output_shape)

class TabMStylePreDecoder(nn.Module):
    """
    TabM-style tensor predecoder.

    This is a tensor-compatible adaptation inspired by TabM-style
    multi-head/ensemble MLPs for tabular prediction.

    It treats each QEC sample as a flattened spatiotemporal tensor:

        Input:  x [B, C_in, T, H, W]
        Output: y [B, C_out, T, H, W]

    Design:
        - shared MLP backbone
        - K lightweight ensemble heads
        - average logits across heads

    Note:
        This is not the official TabM implementation. It is a controlled
        TabM-style baseline adapted to the Ising-Decoding tensor-to-tensor task.
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        width=256,
        depth=3,
        num_heads=8,
        dropout=0.1,
        head_hidden_dim=None,
        debug_shapes=True,
    ):
        super().__init__()

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.width = int(width)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.debug_shapes = debug_shapes

        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)

        if head_hidden_dim is None:
            head_hidden_dim = self.width
        self.head_hidden_dim = int(head_hidden_dim)

        # Shared backbone
        layers = []
        dim = input_dim

        for layer_idx in range(self.depth):
            layers.extend(
                [
                    nn.Linear(dim, self.width),
                    nn.LayerNorm(self.width),
                    nn.GELU(),
                ]
            )
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            dim = self.width

        self.backbone = nn.Sequential(*layers)

        # Head-specific small adapters.
        # This makes the ensemble heads more expressive than simply using K
        # independent linear projections from the same backbone representation.
        self.head_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.width, self.head_hidden_dim),
                    nn.LayerNorm(self.head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
                    nn.Linear(self.head_hidden_dim, output_dim),
                )
                for _ in range(self.num_heads)
            ]
        )

        # Learnable ensemble logits. Softmax gives stable weighted averaging.
        self.ensemble_logits = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, x):
        b = x.shape[0]

        if self.debug_shapes and not hasattr(self, "_debug_printed"):
            print("[TabMStylePreDecoder] input:", tuple(x.shape))
            print("[TabMStylePreDecoder] input_shape:", self.input_shape)
            print("[TabMStylePreDecoder] output_shape:", self.output_shape)
            print("[TabMStylePreDecoder] width:", self.width)
            print("[TabMStylePreDecoder] depth:", self.depth)
            print("[TabMStylePreDecoder] num_heads:", self.num_heads)
            self._debug_printed = True

        z = x.reshape(b, -1)
        z = self.backbone(z)

        # [B, K, output_dim]
        head_outputs = torch.stack([head(z) for head in self.head_adapters], dim=1)

        # [K] -> [1, K, 1]
        weights = torch.softmax(self.ensemble_logits, dim=0).view(1, self.num_heads, 1)

        # Weighted average logits: [B, output_dim]
        y = (head_outputs * weights).sum(dim=1)

        return y.reshape(b, *self.output_shape)

# class TabRStylePreDecoder(nn.Module):
#     """
#     TabR-inspired memory/retrieval tensor predecoder.

#     This is not the original TabR implementation. It adapts the retrieval idea
#     to the Ising-Decoding tensor-to-tensor interface using a learnable memory bank.

#     Input:
#         x: [B, C_in, T, H, W]
#     Output:
#         y: [B, C_out, T, H, W]
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         embed_dim=128,
#         memory_size=256,
#         dropout=0.1,
#         debug_shapes=True,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.embed_dim = embed_dim
#         self.memory_size = memory_size
#         self.debug_shapes = debug_shapes

#         self.token_proj = nn.Sequential(
#             nn.Linear(in_channels, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#         self.query_proj = nn.Linear(embed_dim, embed_dim)

#         self.memory_keys = nn.Parameter(
#             torch.randn(memory_size, embed_dim) / math.sqrt(embed_dim)
#         )
#         self.memory_values = nn.Parameter(
#             torch.randn(memory_size, embed_dim) / math.sqrt(embed_dim)
#         )

#         self.out = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim, out_channels),
#         )

#     def forward(self, x):
#         b, c, t, h, w = x.shape

#         if self.debug_shapes and not hasattr(self, "_debug_printed"):
#             print("[TabRStylePreDecoder] input:", tuple(x.shape))
#             print("[TabRStylePreDecoder] in_channels:", self.in_channels)
#             print("[TabRStylePreDecoder] out_channels:", self.out_channels)
#             print("[TabRStylePreDecoder] memory_size:", self.memory_size)
#             self._debug_printed = True

#         # [B, C, T, H, W] -> [B, T, H, W, C] -> [B, N, C]
#         z = x.permute(0, 2, 3, 4, 1).contiguous()
#         z = z.reshape(b, t * h * w, c)

#         # [B, N, D]
#         z = self.token_proj(z)
#         q = self.query_proj(z)

#         # [B, N, M]
#         attn_logits = torch.matmul(q, self.memory_keys.t()) / math.sqrt(self.embed_dim)
#         attn = torch.softmax(attn_logits, dim=-1)

#         # [B, N, D]
#         retrieved = torch.matmul(attn, self.memory_values)

#         z = z + retrieved

#         # [B, N, C_out]
#         y = self.out(z)

#         # [B, T, H, W, C_out] -> [B, C_out, T, H, W]
#         y = y.reshape(b, t, h, w, self.out_channels)
#         y = y.permute(0, 4, 1, 2, 3).contiguous()

#         return y

class TabRStylePreDecoder(nn.Module):
    """
    TabR-style retrieval tensor predecoder.

    This is a tensor-compatible adaptation inspired by retrieval-based tabular
    models. It treats each spatiotemporal position as a token and retrieves from
    a learnable memory bank.

        Input:  x [B, C_in, T, H, W]
        Output: y [B, C_out, T, H, W]

    Design:
        - token projection from C_in to embed_dim
        - learnable memory keys / values
        - top-k retrieval
        - residual retrieval fusion
        - token-wise output projection

    Note:
        This is not the official TabR implementation. It is a controlled
        TabR-style baseline adapted to QEC syndrome-to-correction prediction.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim=128,
        memory_size=256,
        top_k=16,
        dropout=0.1,
        mlp_ratio=2.0,
        debug_shapes=True,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.memory_size = int(memory_size)
        self.top_k = int(top_k)
        self.dropout = float(dropout)
        self.mlp_ratio = float(mlp_ratio)
        self.debug_shapes = debug_shapes

        self.top_k = max(1, min(self.top_k, self.memory_size))

        self.token_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )

        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.memory_keys = nn.Parameter(
            torch.randn(self.memory_size, self.embed_dim) / math.sqrt(self.embed_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.memory_size, self.embed_dim) / math.sqrt(self.embed_dim)
        )

        self.retrieval_norm = nn.LayerNorm(self.embed_dim)

        hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.out_channels),
        )

    def forward(self, x):
        b, c, t, h, w = x.shape

        if self.debug_shapes and not hasattr(self, "_debug_printed"):
            print("[TabRStylePreDecoder] input:", tuple(x.shape))
            print("[TabRStylePreDecoder] in_channels:", self.in_channels)
            print("[TabRStylePreDecoder] out_channels:", self.out_channels)
            print("[TabRStylePreDecoder] embed_dim:", self.embed_dim)
            print("[TabRStylePreDecoder] memory_size:", self.memory_size)
            print("[TabRStylePreDecoder] top_k:", self.top_k)
            self._debug_printed = True

        # [B, C, T, H, W] -> [B, T, H, W, C] -> [B, N, C]
        z = x.permute(0, 2, 3, 4, 1).contiguous()
        z = z.reshape(b, t * h * w, c)

        # [B, N, D]
        z = self.token_proj(z)

        # Query: [B, N, D]
        q = self.query_proj(z)

        # Normalize for stable retrieval scores
        q_norm = torch.nn.functional.normalize(q, dim=-1)
        k_norm = torch.nn.functional.normalize(self.memory_keys, dim=-1)

        # [B, N, M]
        scores = torch.matmul(q_norm, k_norm.t())

        # Top-k retrieval: [B, N, K]
        top_scores, top_idx = torch.topk(scores, k=self.top_k, dim=-1)

        attn = torch.softmax(top_scores / math.sqrt(self.embed_dim), dim=-1)

        # Gather memory values.
        # memory_values: [M, D]
        # top_idx: [B, N, K]
        # selected_values: [B, N, K, D]
        selected_values = self.memory_values[top_idx]

        # retrieved: [B, N, D]
        retrieved = torch.sum(attn.unsqueeze(-1) * selected_values, dim=-2)

        # Residual retrieval fusion
        z = z + self.retrieval_norm(retrieved)

        # Token-wise FFN residual
        z = z + self.ffn(z)

        # [B, N, C_out]
        y = self.out(z)

        # [B, T, H, W, C_out] -> [B, C_out, T, H, W]
        y = y.reshape(b, t, h, w, self.out_channels)
        y = y.permute(0, 4, 1, 2, 3).contiguous()

        return y

def infer_tensor_shapes_from_cfg(cfg):
    """
    Shape inference for custom models.

    Expected external interface:
        input  [B, C_in,  T, H, W]
        output [B, C_out, T, H, W]

    By default, inherit C_in/C_out from the official public model config:
        cfg.model.input_channels
        cfg.model.out_channels

    You can override them using:
        custom_in_channels
        custom_out_channels
        custom_height
        custom_width_spatial
    """
    distance = int(cfg.distance)
    n_rounds = int(cfg.n_rounds)

    # Prefer explicit custom settings; otherwise inherit official public model channels.
    default_c_in = int(getattr(getattr(cfg, "model", {}), "input_channels", 1))
    default_c_out = int(getattr(getattr(cfg, "model", {}), "out_channels", 1))

    c_in = int(getattr(cfg, "custom_in_channels", default_c_in))
    c_out = int(getattr(cfg, "custom_out_channels", default_c_out))

    # Spatial (H, W) comes from the code family (surface -> square DxD); explicit
    # custom_height / custom_width_spatial overrides still win when provided.
    from qec.code_family import get_code_family

    default_height, default_width = get_code_family(
        getattr(cfg, "code", "surface")
    ).spatial_shape(distance)
    height = int(getattr(cfg, "custom_height", default_height))
    width = int(getattr(cfg, "custom_width_spatial", default_width))

    input_shape = (c_in, n_rounds, height, width)
    output_shape = (c_out, n_rounds, height, width)

    return input_shape, output_shape, c_in, c_out

