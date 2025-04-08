import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import pennylane as qml

# Define quantum devices and number of qubits
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit for feature transformation
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode classical data into quantum state
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum circuit
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure in computational basis
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Quantum layer that can be integrated with PyTorch
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(shape) * 0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, -1)
        
        # Process each sample in the batch
        result = torch.zeros((batch_size, self.n_qubits), device=x.device)
        for i in range(batch_size):
            # Select n_qubits features for quantum processing
            inputs = x_reshaped[i, :self.n_qubits].detach().cpu().numpy()
            # Run quantum circuit
            quantum_output = torch.tensor(quantum_circuit(inputs, self.weights.detach().cpu().numpy()))
            result[i] = quantum_output
            
        return result

# Quantum attention mechanism
class QuantumAttention(nn.Module):
    def __init__(self, d_model, n_qubits=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        
        # Linear layers to project to quantum space
        self.q_linear = nn.Linear(d_model, n_qubits)
        self.k_linear = nn.Linear(d_model, n_qubits)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Quantum layers
        self.q_quantum = QuantumLayer(n_qubits, n_layers)
        self.k_quantum = QuantumLayer(n_qubits, n_layers)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Project to quantum space
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # Apply quantum transformation
        q_quantum = self.q_quantum(q).unsqueeze(1).expand(-1, seq_len, -1)
        k_quantum = self.k_quantum(k)
        
        # Calculate attention scores with quantum features
        scores = torch.bmm(q_quantum, k_quantum.transpose(1, 2)) / np.sqrt(self.n_qubits)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v)
        out = self.out_proj(out)
        
        return out

def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

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
            index = i*self.stride
            input = x[:, :, index:index+self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)

        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, use_quantum=False):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        self.use_quantum = use_quantum

        if use_quantum:
            self.quantum_attention = QuantumAttention(d_model, n_qubits=4, n_layers=2)
        else:
            self.w_q = nn.Linear(d_model, n_head*self.d_k)
            self.w_k = nn.Linear(d_model, n_head*self.d_k)
            self.w_v = nn.Linear(d_model, n_head*self.d_v)
            self.w_o = nn.Linear(n_head*self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    # [batch_size, n_channel, d_model]
    def forward(self, query, key, value):
        if self.use_quantum:
            return self.quantum_attention(query, key, value)
        
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out

class QuantumFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout, n_qubits=4, n_layers=2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        self.proj = nn.Linear(n_qubits, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = self.w_1(x)
        x = self.act(x)
        
        # Process sequence dimension with quantum layer
        x_flat = x.reshape(-1, x.size(-1))
        q_features = self.quantum_layer(x_flat[:, :n_qubits])
        q_features = q_features.reshape(batch_size, seq_len, -1)
        
        # Project quantum features back to original dimension
        x = self.proj(q_features)
        x = self.dropout(x)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout, use_quantum=False):
        super().__init__()
        self.use_quantum = use_quantum
        
        if use_quantum:
            self.quantum_ff = QuantumFeedForward(d_model, d_hidden, dropout)
        else:
            self.w_1 = nn.Linear(d_model, d_hidden)
            self.act = nn.GELU()
            self.w_2 = nn.Linear(d_hidden, d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_quantum:
            return self.quantum_ff(x)
            
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5, use_quantum=False):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop, use_quantum=use_quantum)
        self.feed_forward = FeedForward(embed_dim, embed_dim*fc_ratio, fc_drop, use_quantum=use_quantum)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

class QuantumTransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, 
    pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5, quantum_layers=[0, 2]):
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

        self.dropout = nn.Dropout()

        # Create transformer encoders with quantum layers at specified positions
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim, 
                num_heads, 
                fc_ratio, 
                attn_drop, 
                fc_drop, 
                use_quantum=(i in quantum_layers)
            ) for i in range(depth)
        ])

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(122, 64, (2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        # Add quantum layer for final classification
        self.pre_classify = nn.Linear(embed_dim*temp_embedding_dim, 16)
        self.quantum_classify = QuantumLayer(n_qubits=4, n_layers=3)
        self.post_classify = nn.Linear(4, num_classes)

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
        
        # Quantum-enhanced classification
        x = self.pre_classify(x)
        q_features = self.quantum_classify(x)
        out = self.post_classify(q_features)

        return out

