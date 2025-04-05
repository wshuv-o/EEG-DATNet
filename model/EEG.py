import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def attention(query, key, value):
    """Advanced attention mechanism for neural feature integration"""
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn


class EnhancedVariancePooling(nn.Module):
    """Advanced variance-based pooling with logarithmic normalization"""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i*self.stride
            input = x[:, :, index:index+self.kernel_size]
            # Enhanced variance calculation with improved numerical stability
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)
        return out

class AdvancedMultiHeadAttention(nn.Module):
    """Multi-head attention with improved feature projection"""
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        # Enhanced projection matrices
        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Advanced tensor reshaping for multi-head processing
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out

class AdvancedFeedForward(nn.Module):
    """Enhanced feed-forward network with GELU activation"""
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        # Using GELU for improved gradient flow
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Two-stage processing with regularization
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x
# Define Spiking Activation Layer
class SpikingActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        valid = (input >= ctx.threshold - 0.5) & (input <= ctx.threshold + 0.5)
        return grad_input * valid.float(), None

class SpikingActivationLayer(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return SpikingActivation.apply(x, self.threshold)
    
class EnhancedTransformerBlock(nn.Module):
    """Advanced transformer block with pre-normalization architecture"""
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = AdvancedMultiHeadAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = AdvancedFeedForward(embed_dim, embed_dim*fc_ratio, fc_drop)
        # Using layer normalization for improved training stability
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, data):
        # Pre-normalization architecture for improved gradient flow
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output
class AttentionLayer(nn.Module):
    """Self-attention layer for temporal feature emphasis"""
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = np.sqrt(input_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        q = self.query(x)  # [batch, seq_len, features]
        k = self.key(x)    # [batch, seq_len, features]
        v = self.value(x)  # [batch, seq_len, features]
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [batch, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=2)  # [batch, seq_len, seq_len]
        
        # Apply attention weights
        context = torch.bmm(attn_weights, v)  # [batch, seq_len, features]
        return context
class NeuroTransNet(nn.Module):
    """
    Advanced EEG analysis model with multi-scale temporal processing,
    dual-path feature extraction, and transformer-based integration.
    
    This model employs a novel architecture specifically designed for
    EEG signal processing with enhanced feature extraction capabilities.
    """
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, 
    pool_stride=15, num_heads=8, fc_ratio=4, depth=4, lstm_hidden=64, lstm_layers=2,spike_threshold=0.50 , attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        # Multi-scale temporal convolutions for comprehensive feature extraction
        self.temporal_conv_small = nn.Conv2d(1, embed_dim//4, (1, 15), padding=(0, 7))
        self.temporal_conv_medium = nn.Conv2d(1, embed_dim//4, (1, 25), padding=(0, 12))
        self.temporal_conv_large = nn.Conv2d(1, embed_dim//4, (1, 51), padding=(0, 25))
        self.temporal_conv_xlarge = nn.Conv2d(1, embed_dim//4, (1, 65), padding=(0, 32))
        
        # Normalization for improved training stability
        self.batch_norm1 = nn.BatchNorm2d(embed_dim)
        self.spike_act1 = SpikingActivationLayer(spike_threshold)

        # Spatial feature extraction
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))
        self.batch_norm2 = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ELU()
        self.spike_act2 = SpikingActivationLayer(spike_threshold)

        # Dual-path feature pooling
        self.variance_pooling = EnhancedVariancePooling(pool_size, pool_stride)
        self.average_pooling = nn.AvgPool1d(pool_size, pool_stride)

        # Calculate embedding dimension
        self.temporal_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        # Regularization
        self.dropout = nn.Dropout()

        # Transformer-based feature integration
        # self.transformer_blocks = nn.ModuleList(
        #     [EnhancedTransformerBlock(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        # )

        # # Feature encoding and classification
        # self.feature_encoder = nn.Sequential(
        #     nn.Conv2d(122, 64, (2, 1)),
        #     nn.BatchNorm2d(64),
        #     nn.ELU()
        # )
                # Bi-LSTM (bidirectional so output features are 2 * lstm_hidden)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, 
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        # The two branches (avg and var) are concatenated:
        # Flattened feature vector size = (lstm_hidden * 2) * temp_embedding_dim * 2
        self.fc = nn.Linear(31232, num_classes)

        # # Final classification layer
        # self.classifier = nn.Linear(embed_dim*self.temporal_embedding_dim, num_classes)
        
        # Initialize weights for optimal convergence
    #     self._initialize_weights()
    
    # def _initialize_weights(self):
    #     """Advanced weight initialization for improved convergence"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input preparation
        x = x.unsqueeze(dim=1)  # [batch, 1, channels, samples]
        
        # Multi-scale temporal feature extraction
        x1 = self.temporal_conv_small(x)
        x2 = self.temporal_conv_medium(x)
        x3 = self.temporal_conv_large(x)
        x4 = self.temporal_conv_xlarge(x)
        
        # Feature fusion
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.batch_norm1(x)
        x = self.spike_act1(x)
        
        # Spatial feature extraction
        x = self.spatial_conv(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.spike_act2(x)
        x = x.squeeze(dim=2)  # [batch, embed_dim, samples]

        # Dual-path feature pooling
        x1 = self.average_pooling(x)  # Mean statistics
        x2 = self.variance_pooling(x)  # Variance statistics

        # Regularization
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        # Prepare for transformer processing
        x1 = rearrange(x1, 'b d n -> b n d')  # [batch, seq_len, features]
        x2 = rearrange(x2, 'b d n -> b n d')

        # Apply transformer blocks for feature integration
        # for transformer in self.transformer_blocks:
        #     x1 = transformer(x1)
        #     x2 = transformer(x2)
                # Pass through LSTM (each branch independently)
        x1, _ = self.lstm(x1)  # -> (batch, temp_embedding_dim, 2*lstm_hidden)
        x2, _ = self.lstm(x2)  # -> (batch, temp_embedding_dim, 2*lstm_hidden)

        # Prepare for feature encoding
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)

        # Feature fusion
        x = torch.cat((x1, x2), dim=2)
        # x = self.feature_encoder(x)

        # Final classification
        x = x.reshape(x.size(0), -1)
        print(f"Shape before fc: {x.shape}")
        out = self.fc(x)

        return out
