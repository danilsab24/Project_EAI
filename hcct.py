import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=192):
        
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class Conv3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(1,   32),   # 192 -> 96
            ConvBlock(32,  64),   # 96 -> 48
            ConvBlock(64, 128),   # 48 -> 24
            ConvBlock(128,256),   # 24 -> 12
            ConvBlock(256,512),   # 12 -> 6 
        )

        self.num_patches = 6 * 6 * 6
        self.hidden_size = 512
        
    def forward(self, x):
        # x.shape = [B, C, D, H, W]
        x = self.blocks(x)   # [B, 512, 6, 6, 6]
        B, C, D, H, W = x.shape
        # each filter produces a 6x6x6 patch
        x = x.view(B, C, -1)  # → [B, 512, 216]
        return x.permute(0, 2, 1)  # [B, 216, 512]

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config["embed_dim"]

        # Layer che trasforma l’immagine in patch embeddings
        self.patch_embed = Conv3DEncoder()
        num_patches = self.patch_embed.num_patches

        # Token [CLS] (1, 1, H)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )

        # Positional embeddings (1, 1 + num_patches, H)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, self.embed_dim)
        )

        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        # x.shape = [B, C, D, H, W]
        
        # Patch embeddings
        # (batch_size, num_patches, embed_dim)
        x = self.patch_embed(x)
        B = x.size(0)

        # Replica il CLS token sul batch
        # (batch_size, 1, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)

        # Prepend CLS al sequence dei patch
        # (batch_size, 1 + num_patches, embed_dim)
        x = torch.cat((cls, x), dim=1)

        # Aggiungi posizioni e applica dropout
        x = x + self.pos_embed
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=True) # W_q
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=True)   # W_k
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=True) # W_v 
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # W_o
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : [b, seq_length, embed_dim]
        # where seq_length = num_patches + 1
        b, seq_length, embed_dim = x.shape
        
        #  NOTE: X * W_q^T is a (b x seq_length x embed_dim) x (b x embed_dim x q)
        # (b, seq_length, embed_dim)
        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # NOTE: self.embed_dim = num_head * head_dim 
        # We split the matrix by adding a `num_heads` dimension
        # (b, seq_length, embed_dim) -> (b, seq_length, num_heads, head_dim)
        keys = keys.view(b, seq_length, self.num_heads, self.head_dim)
        values = values.view(b, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(b, seq_length, self.num_heads, self.head_dim)

        # (b, seq_length, num_heads, head_dim) -> (b, num_heads, seq_length, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # NOTE: Dot product for each head (Q*K^T)

        # NOTE: H = softmax(Q*K^T * M /sqrt(q))*V
        # shape = (b, num_heads, seq_length, seq_length)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # NOTE: Dropout: H = Mask * H / (1 - drop_raio)   
        attn_weights = self.dropout(attn_weights)
        
        # (attn_weights @ values).shape: [(b, num_heads, seq_length, seq_length) * (b, num_heads, num_patches, head_dim)] = (b, num_heads, num_patches, head_dim)
        # Shape: (b, num_heads, seq_length, head_dim) -> (b, seq_length, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.embed_dim = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, seq_length, self.embed_dim)
        # NOTE: [H_1, ... , H_h] * W_o
        # shape: [(b, seq_length, self.embed_dim) * (self.embed_dim * self.embed_dim)] = [b, num_patches, self.embed_dim]
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec # shape: [b, seq_length, self.embed_dim]

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["dropout_rate"])
        
        self.ff = FeedForward(cfg["embed_dim"])
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        # x.shape = [batch_size, num_patches + 1, embed_dim]
        shortcut = x
        x = self.norm1(x)
        # x.shape = [batch_size, num_patches + 1, embed_dim]
        x = self.att(x) 
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        # [batch_size, num_patches + 1, embed_dim] -> [batch_size, num_patches + 1, embed_dim]
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x # x.shape = [batch_size, num_patches + 1, embed_dim]

class HybridPooling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.g = nn.Linear(cfg["embed_dim"], 1)
    
    def forward(self, x_L):
        # x_L.shape = [batch_size, num_patches + 1, embed_dim]
        # xc.shape = [batch_size, 1, embed_dim]
        xc = x_L[:, 0:1, :]
        # xa.shape = [batch_size, num_patches, embed_dim]
        xa = x_L[:, 1:, :]

        # scores.shape = [batch_size, num_patches, 1]
        scores = self.g(xa)
        # [batch_size, num_patches, 1] -> [batch_size, 1, num_patches]
        scores = scores.transpose(1, 2) 
        # x_prime_a.shape = [batch_size, 1, num_patches]
        x_prime_a = F.softmax(scores, dim=-1)

        # (b, 1, n) @ (b, n, d) -> (b, 1, d)
        x_double_prime_a = x_prime_a @ xa

        # z.shape = [batch_size, 1, 2 * num_patches]
        z = torch.cat([xc, x_double_prime_a], dim=-1)
        return z 

class HCCT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd = Embeddings(config)
        self.trf_blocks = nn.Sequential(
            *[TransformerEncoder(config) for _ in range(config["n_layers"])])
        
        self.pooling = HybridPooling(config)
        self.classifier = nn.Linear(2 * config["embed_dim"] , config["num_classes"])

    def forward(self, x):
        # x.shape = [batch_size, C, D, H, W]
        # [batch_size, C, D, H, W] -> [batch_size, num_patches + 1, embed_dim]
        x = self.embd(x)
        # [batch_size, num_patches + 1, embed_dim] -> [batch_size, num_patches + 1, embed_dim]
        for trf_block in self.trf_blocks:
            x = trf_block(x)
        # [batch_size, num_patches + 1, embed_dim] -> [batch_size, 1, 2 * num_patches]
        x = self.pooling(x)
        # [batch_size, 1, 2 * num_patches] -> [batch_size, 2 * num_patches]
        x = torch.squeeze(x, dim=1)
        # [batch_size, 2 * num_patches] -> [batch_size, num_classes]
        x = self.classifier(x)
        return x

