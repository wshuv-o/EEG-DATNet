import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Attention function
def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**0.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

# Variable Pooling (Variance Pooling) Layer
class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)
        return out

# Dual Attention Mechanism
class DualAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super(DualAttention, self).__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model

        # Query, Key, and Value transformations
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        
        # Spatial attention - operates on sequence dimension
        self.spatial_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Channel attention - operates on feature dimension
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Apply transformations for multi-head attention
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        # Debug shapes
        # print("Shape of q:", q.shape)
        # print("Shape of k:", k.shape)
        # print("Shape of v:", v.shape)
        
        # Apply spatial attention (across sequence dimension)
        # First, reshape v to apply attention properly
        v_spatial = rearrange(v, 'b h n d -> b n (h d)')
        
        # Get spatial attention weights
        spatial_weights = self.spatial_attention(v_spatial)  # shape: [b, n, d_model]
        
        # Apply spatial attention and reshape back
        v_spatial = v_spatial * spatial_weights
        v_spatial = rearrange(v_spatial, 'b n (h d) -> b h n d', h=self.n_head)
        
        # Apply channel attention (across feature dimension)
        # Average across sequence dimension to get channel profile
        channel_profile = value.mean(dim=1, keepdim=True)  # shape: [b, 1, d_model]
        
        # Get channel attention weights
        channel_weights = self.channel_attention(channel_profile)  # shape: [b, 1, d_model]
        
        # Expand to match sequence dimension
        channel_weights = channel_weights.expand(-1, seq_len, -1)  # shape: [b, n, d_model]
        
        # Apply channel attention to value and reshape
        v_channel = value * channel_weights
        v_channel = rearrange(self.w_v(v_channel), 'b n (h d) -> b h n d', h=self.n_head)
        
        # Combine both attentions (spatial and channel)
        v_dual = v_spatial + v_channel
        
        # Apply standard attention mechanism
        out, _ = attention(q, k, v_dual)
        
        # Reshape and apply final projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.dropout(self.w_o(out))
        
        return out
# Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.dual_attention = DualAttention(embed_dim, num_heads, attn_drop)  # Replace multi-head attention with dual attention
        self.feed_forward = FeedForward(embed_dim, embed_dim*fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.dual_attention(res, res, res)  # Apply dual attention

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

# Final TransNet Model
class NeuroTransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, 
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim//4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, embed_dim//4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, embed_dim//4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, embed_dim//4, (1, 65), padding=(0, 32))
        
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))

        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout(0.6)

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(122, 64, (2, 1)),  # Change input channels from 64 to 122
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.classify = nn.Linear(embed_dim*temp_embedding_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)
        x = torch.cat((x1,x2,x3,x4), dim=1)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = x.squeeze(dim=2)

        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')

        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)
        
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)

        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)

        out = self.classify(x)

        return out
