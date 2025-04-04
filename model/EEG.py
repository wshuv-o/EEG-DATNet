import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import torch.fft as fft

def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

class CrossAttention(nn.Module):
    """Cross-attention module for integrating information between time and frequency domains"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        
        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))
        
        return out

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class SpectralTransform(nn.Module):
    """Transform time-domain signals to frequency domain and process frequency bands"""
    def __init__(self, in_channels, out_channels, num_channels, n_fft=256, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_channels = num_channels
        
        # Number of frequency bins in STFT
        self.freq_bins = n_fft // 2 + 1
        
        # Convolutional layers for frequency domain processing
        self.freq_conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, out_channels // 2, kernel_size=3, padding=1),  # Real and imaginary parts
            nn.BatchNorm2d(out_channels // 2),
            nn.ELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        
        # Frequency attention
        self.freq_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, time_len = x.size()
        
        # Reshape for STFT
        x = x.view(batch_size * channels, time_len)
        
        # Apply STFT
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            return_complex=True,
            normalized=True
        )
        
        # Convert complex tensor to real tensor with shape [batch*channels, 2, freq_bins, time_frames]
        stft_real = torch.view_as_real(stft)  # [batch*channels, freq_bins, time_frames, 2]
        stft_real = stft_real.permute(0, 3, 1, 2)  # [batch*channels, 2, freq_bins, time_frames]
        
        # Reshape to [batch, channels*2, freq_bins, time_frames]
        time_frames = stft_real.size(3)
        
        # Use reshape instead of view to handle non-contiguous tensors
        stft_real = stft_real.reshape(batch_size, channels * 2, self.freq_bins, time_frames)
        
        # Apply frequency domain processing
        freq_features = self.freq_conv(stft_real)
        
        # Apply frequency attention
        attention_weights = self.freq_attention(freq_features)
        freq_features = freq_features * attention_weights
        
        # Global average pooling across frequency dimension
        freq_features = freq_features.mean(dim=2)  # [batch, out_channels, time_frames]
        
        return freq_features



class DilatedTemporalConv(nn.Module):
    """Multi-scale dilated convolutions for temporal processing"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        
        # Dilated convolutions with different dilation rates
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, (1, 15), padding=(0, 7), dilation=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, (1, 15), padding=(0, 14), dilation=(1, 2))
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, (1, 15), padding=(0, 28), dilation=(1, 4))
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, (1, 15), padding=(0, 56), dilation=(1, 8))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn(x)
        x = self.elu(x)
        
        return x

class AdaptivePooling(nn.Module):
    def __init__(self, kernel_size, stride, embed_dim):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Learnable weights for different pooling operations
        self.pool_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Projection layer
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # Adjust kernel size and stride if input is too small
        t = x.shape[2]
        kernel_size = min(self.kernel_size, t // 2)  # Ensure kernel size is at most half the input size
        stride = min(self.stride, kernel_size)       # Ensure stride is not larger than kernel size
        
        # Create pooling operations with adjusted parameters
        avg_pool = nn.AvgPool1d(kernel_size, stride)
        max_pool = nn.MaxPool1d(kernel_size, stride)
        
        # Apply different pooling operations
        avg_pooled = avg_pool(x)
        max_pooled = max_pool(x)
        
        # Custom var pooling with adjusted parameters
        out_shape = (t - kernel_size) // stride + 1
        var_pooled = []
        for i in range(out_shape):
            index = i * stride
            input = x[:, :, index:index+kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            var_pooled.append(output)
        var_pooled = torch.cat(var_pooled, dim=-1)
        
        # Apply softmax to weights
        weights = F.softmax(self.pool_weights, dim=0)
        
        # Weighted sum of pooling results
        pooled = weights[0] * avg_pooled + weights[1] * max_pooled + weights[2] * var_pooled
        
        # Apply projection
        pooled = rearrange(pooled, 'b d n -> b n d')
        pooled = self.proj(pooled)
        pooled = rearrange(pooled, 'b n d -> b d n')
        
        return pooled

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    # [batch_size, n_channel, d_model]
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out

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

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim*fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

class DualStreamEEGNet(nn.Module):
    """
    Novel EEG analysis model with dual-stream processing (time and frequency domains),
    squeeze-and-excitation blocks, dilated convolutions, and cross-attention mechanisms.
    
    Data shape: [batch_size, 22 channels, 1875 datapoints]
    """
    def __init__(self, num_classes=4, num_samples=1875, num_channels=22, embed_dim=32, 
                 pool_size=50, pool_stride=15, num_heads=8, fc_ratio=4, depth=4, 
                 attn_drop=0.5, fc_drop=0.5, n_fft=256, hop_length=128):
        super().__init__()
        
        # Parameters
        self.embed_dim = embed_dim
        self.num_samples = num_samples
        self.num_channels = num_channels
        
        # Time-domain stream
        self.dilated_conv = DilatedTemporalConv(1, embed_dim)
        self.se_block = SqueezeExcitation(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))
        self.bn_time = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()
        
        # Frequency-domain stream
        #self.spectral_transform = SpectralTransform(1, embed_dim, n_fft, hop_length)
        # Frequency-domain stream
        # Frequency-domain stream
        self.spectral_transform = SpectralTransform(1, embed_dim, num_channels, n_fft, hop_length)

        # Calculate time frames after STFT
        self.time_frames = (num_samples - n_fft) // hop_length + 1
        
        # Adaptive pooling
        self.adaptive_pool = AdaptivePooling(pool_size, pool_stride, embed_dim)
        
        # Calculate temporal embedding dimension after pooling
        self.temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.freq_embedding_dim = (self.time_frames - pool_size) // pool_stride + 1
        
        # Cross-attention for domain integration
        self.cross_attention = CrossAttention(embed_dim, num_heads, attn_drop)
        
        # Transformer encoders
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)]
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Final processing
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(122, 64, (2, 1)),  # Use the actual number of channels from your input
            nn.BatchNorm2d(64),
            nn.ELU(),
            # nn.Conv2d(64, 32, (1, 1)),
            # nn.BatchNorm2d(32),
            # nn.ELU()
        )
        
        # Classification head
        # self.classify = nn.Sequential(
        #     nn.Linear(embed_dim * max(self.temp_embedding_dim, self.freq_embedding_dim), 
        #              embed_dim * 4),
        #     nn.Dropout(0.5),
        #     nn.ELU(),
        #     nn.Linear(embed_dim * 4, num_classes)
        # )
        self.classify = nn.Linear(embed_dim*self.temp_embedding_dim, num_classes)
 
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Time-domain stream
        x_time = x.unsqueeze(dim=1)  # [batch, 1, channels, samples]
        x_time = self.dilated_conv(x_time)  # [batch, embed_dim, channels, samples]
        x_time = self.se_block(x_time)  # Apply channel recalibration
        x_time = self.spatial_conv(x_time)  # [batch, embed_dim, 1, samples]
        x_time = self.bn_time(x_time)
        x_time = self.elu(x_time)
        x_time = x_time.squeeze(dim=2)  # [batch, embed_dim, samples]
        
        # Frequency-domain stream
        x_freq = self.spectral_transform(x)  # [batch, embed_dim, time_frames]
        
        # Apply adaptive pooling to both streams
        x_time_pooled = self.adaptive_pool(x_time)  # [batch, embed_dim, temp_embedding_dim]
        x_freq_pooled = self.adaptive_pool(x_freq)  # [batch, embed_dim, freq_embedding_dim]
        
        # Ensure same temporal dimension for cross-attention
        # Pad or truncate as needed
        target_len = max(x_time_pooled.size(2), x_freq_pooled.size(2))
        
        if x_time_pooled.size(2) < target_len:
            padding = target_len - x_time_pooled.size(2)
            x_time_pooled = F.pad(x_time_pooled, (0, padding))
        else:
            x_time_pooled = x_time_pooled[:, :, :target_len]
            
        if x_freq_pooled.size(2) < target_len:
            padding = target_len - x_freq_pooled.size(2)
            x_freq_pooled = F.pad(x_freq_pooled, (0, padding))
        else:
            x_freq_pooled = x_freq_pooled[:, :, :target_len]
        
        # Rearrange for transformer processing
        x_time_pooled = rearrange(x_time_pooled, 'b d n -> b n d')  # [batch, temp_embedding_dim, embed_dim]
        x_freq_pooled = rearrange(x_freq_pooled, 'b d n -> b n d')  # [batch, freq_embedding_dim, embed_dim]
        
        # Apply cross-attention for domain integration
        x_time_attended = x_time_pooled + self.cross_attention(x_time_pooled, x_freq_pooled, x_freq_pooled)
        x_freq_attended = x_freq_pooled + self.cross_attention(x_freq_pooled, x_time_pooled, x_time_pooled)
        
        # Apply transformer encoders
        for encoder in self.transformer_encoders:
            x_time_attended = encoder(x_time_attended)
            x_freq_attended = encoder(x_freq_attended)
        
        # Reshape for final processing
        x_time_final = x_time_attended.unsqueeze(dim=2)  # [batch, temp_embedding_dim, 1, embed_dim]
        x_freq_final = x_freq_attended.unsqueeze(dim=2)  # [batch, freq_embedding_dim, 1, embed_dim]
        
        # Concatenate along channel dimension
        x_combined = torch.cat((x_time_final, x_freq_final), dim=2)  # [batch, temp_embedding_dim, 2, embed_dim]
        
        # Apply final convolutional encoding
        x_encoded = self.conv_encoder(x_combined)  # [batch, temp_embedding_dim, 1, embed_dim]
        
        # Reshape for classification
        x_flat = x_encoded.reshape(batch_size, -1)  # [batch, temp_embedding_dim * embed_dim]
        
        # Apply classification head
        out = self.classify(x_flat)  # [batch, num_classes]
        
        return out
