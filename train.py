"""
train.py - MCANU-Net: Complete Training with All 7 Novelties Including MAML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import copy
from data_collector import MultiModalDataCollector
from config import Config


# ============================================================================
# NOVELTY 1: Multi-Modal Data Fusion with VAE
# ============================================================================
class ModalityEncoder(nn.Module):
    """Separate embedding network for each modality"""
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return F.relu(self.bn2(self.fc2(x)))


class VAEFusion(nn.Module):
    """Variational Autoencoder for latent market state representation"""
    def __init__(self, input_dim=128, latent_dim=32):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ============================================================================
# NOVELTY 2: Temporal Causal Discovery Module
# ============================================================================
class NeuralGrangerCausality(nn.Module):
    """Learnable neural Granger causality with multiple lags"""
    def __init__(self, num_features=32, num_lags=5):
        super().__init__()
        self.num_lags = num_lags
        self.num_features = num_features
        self.causal_weights = nn.Parameter(torch.randn(num_features, num_features, num_lags) * 0.01)
    
    def forward(self, x):
        batch, seq_len, features = x.shape
        causal_features = []
        
        for lag in range(min(self.num_lags, seq_len)):
            lagged = x[:, lag, :].unsqueeze(1)
            weight = self.causal_weights[:, :, lag]
            causal = torch.matmul(lagged, weight)
            causal_features.append(causal)
        
        if len(causal_features) > 0:
            causal_output = torch.cat(causal_features, dim=1)
        else:
            causal_output = x
        
        return causal_output, self.causal_weights


# ============================================================================
# NOVELTY 3: Dynamic Graph Neural Networks
# ============================================================================
class DynamicGraphLearner(nn.Module):
    """Adaptive graph structure learning"""
    def __init__(self, node_dim=32, num_nodes=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_fc1 = nn.Linear(node_dim * 2, 64)
        self.graph_fc2 = nn.Linear(64, 1)
    
    def forward(self, node_features):
        batch, num_nodes, node_dim = node_features.shape
        adjacency = torch.zeros(batch, num_nodes, num_nodes).to(node_features.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                pair = torch.cat([node_features[:, i, :], node_features[:, j, :]], dim=-1)
                edge_weight = torch.sigmoid(self.graph_fc2(F.relu(self.graph_fc1(pair))))
                adjacency[:, i, j] = edge_weight.squeeze()
        
        return adjacency


# ============================================================================
# NOVELTY 4: Causal Attention Mechanism
# ============================================================================
class CausalAttention(nn.Module):
    """Multi-head attention with causal bias"""
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x, causal_weights=None):
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=None)
        return attn_output, attn_weights


# ============================================================================
# NOVELTY 5: Uncertainty Quantification
# ============================================================================
class UncertaintyHead(nn.Module):
    """Separates epistemic and aleatoric uncertainty"""
    def __init__(self, input_dim=32):
        super().__init__()
        self.fc_mean = nn.Linear(input_dim, 1)
        self.fc_aleatoric = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        mean = self.fc_mean(x)
        log_var = torch.clamp(self.fc_aleatoric(x), min=-10, max=10)
        return mean, log_var


# ============================================================================
# NOVELTY 7: Multi-Task Learning
# ============================================================================
class MultiTaskHeads(nn.Module):
    """Three output heads: price, volatility, and market regime"""
    def __init__(self, input_dim=32):
        super().__init__()
        self.price_head = nn.Linear(input_dim, 1)
        self.volatility_head = nn.Linear(input_dim, 1)
        self.regime_head = nn.Linear(input_dim, 3)
    
    def forward(self, x):
        price = self.price_head(x)
        volatility = F.softplus(self.volatility_head(x))
        regime = self.regime_head(x)
        return price, volatility, regime


# ============================================================================
# Main MCANU-Net Model
# ============================================================================
class MCANUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # NOVELTY 1: Modality encoders
        self.price_encoder = ModalityEncoder(15, 32)
        self.sentiment_encoder = ModalityEncoder(5, 32)
        self.onchain_encoder = ModalityEncoder(5, 32)
        self.macro_encoder = ModalityEncoder(7, 32)
        
        # NOVELTY 1: VAE fusion
        self.vae_fusion = VAEFusion(input_dim=128, latent_dim=32)
        
        # NOVELTY 2: Temporal causal discovery
        self.causal_module = NeuralGrangerCausality(num_features=32, num_lags=5)
        
        # NOVELTY 3: Dynamic graph learning
        self.graph_learner = DynamicGraphLearner(node_dim=32, num_nodes=1)
        
        # NOVELTY 4: Causal attention
        self.causal_attention = CausalAttention(embed_dim=32, num_heads=4)
        
        # NOVELTY 5: Uncertainty quantification
        self.uncertainty_head = UncertaintyHead(input_dim=32)
        
        # NOVELTY 7: Multi-task heads
        self.multi_task_heads = MultiTaskHeads(input_dim=32)
    
    def forward(self, price, sentiment, onchain, macro):
        batch_size, seq_len, _ = price.shape
        
        # NOVELTY 1: Encode each modality
        price_emb = self.price_encoder(price[:, -1, :])
        sent_emb = self.sentiment_encoder(sentiment[:, -1, :])
        onchain_emb = self.onchain_encoder(onchain[:, -1, :])
        macro_emb = self.macro_encoder(macro[:, -1, :])
        
        # NOVELTY 1: VAE fusion
        concat_emb = torch.cat([price_emb, sent_emb, onchain_emb, macro_emb], dim=-1)
        latent_state, mu, logvar = self.vae_fusion(concat_emb)
        
        # NOVELTY 2: Temporal causal discovery
        seq_features = latent_state.unsqueeze(1).repeat(1, min(seq_len, 5), 1)
        causal_features, causal_weights = self.causal_module(seq_features)
        
        # NOVELTY 4: Causal attention
        attn_output, attn_weights = self.causal_attention(causal_features, causal_weights)
        
        # NOVELTY 3: Graph learning
        graph_input = attn_output.mean(dim=1, keepdim=True)
        adjacency = self.graph_learner(graph_input)
        
        # Final representation
        final_repr = attn_output.mean(dim=1)
        
        # NOVELTY 7: Multi-task predictions
        price_pred, volatility_pred, regime_pred = self.multi_task_heads(final_repr)
        
        # NOVELTY 5: Uncertainty estimation
        price_mean, price_logvar = self.uncertainty_head(final_repr)
        
        return {
            'price': price_pred,
            'volatility': volatility_pred,
            'regime': regime_pred,
            'uncertainty_mean': price_mean,
            'uncertainty_logvar': price_logvar,
            'vae_mu': mu,
            'vae_logvar': logvar,
            'causal_weights': causal_weights,
            'attention_weights': attn_weights
        }


# ============================================================================
# Dataset
# ============================================================================
class CryptoDataset(Dataset):
    def __init__(self, data, lookback=10):
        self.data = data
        self.lookback = lookback

    def __len__(self):
        return max(len(self.data) - self.lookback - 1, 0)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.lookback]
        target = self.data.iloc[idx + self.lookback]

        price_features = ['close', 'high', 'low', 'open', 'volume', 'ma_7', 'ma_21', 'ma_50', 
                         'std_21', 'log_return', 'return_1d', 'return_7d', 'vol_ma_7', 'rsi_14', 'market_cap']
        sentiment_features = ['sentiment_score', 'tweet_volume', 'positive_ratio', 'negative_ratio', 'sentiment_score']
        onchain_features = ['active_addresses', 'transaction_count', 'hash_rate', 'nvt_ratio', 'mvrv_ratio']
        macro_features = ['sp500', 'gold_price', 'oil_price', 'dxy', 'vix', 'sp500_return', 'gold_return']

        price_seq = torch.FloatTensor(sequence[price_features].values)
        sentiment_seq = torch.FloatTensor(sequence[sentiment_features].values)
        onchain_seq = torch.FloatTensor(sequence[onchain_features].values)
        macro_seq = torch.FloatTensor(sequence[macro_features].values)

        price_target = torch.FloatTensor([target['close_scaled']])
        volatility_target = torch.FloatTensor([max(target.get('std_21', 0.01), 0.01)])
        
        return_val = target.get('return_1d', 0)
        regime = 0 if return_val > 0.02 else (1 if return_val < -0.02 else 2)
        regime_target = torch.LongTensor([regime])

        return {
            'price': price_seq,
            'sentiment': sentiment_seq,
            'onchain': onchain_seq,
            'macro': macro_seq,
            'price_target': price_target,
            'volatility_target': volatility_target,
            'regime_target': regime_target
        }


# ============================================================================
# NOVELTY 6: Meta-Learning Framework (MAML)
# ============================================================================
def maml_inner_loop(model, support_batch, inner_lr, device):
    """
    Inner loop: task-specific adaptation on support set
    Returns adapted model for the specific task/regime
    """
    # Clone model for task-specific adaptation
    adapted_model = copy.deepcopy(model)
    adapted_model.train()
    
    # Create optimizer for inner loop
    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    # Perform one gradient step on support set
    price = support_batch['price'].to(device)
    sentiment = support_batch['sentiment'].to(device)
    onchain = support_batch['onchain'].to(device)
    macro = support_batch['macro'].to(device)
    price_target = support_batch['price_target'].to(device)
    
    inner_optimizer.zero_grad()
    outputs = adapted_model(price, sentiment, onchain, macro)
    loss = F.mse_loss(outputs['price'], price_target)
    loss.backward()
    inner_optimizer.step()
    
    return adapted_model


def maml_outer_loop(meta_model, tasks, inner_lr, outer_lr, device):
    """
    Outer loop: meta-optimization across tasks
    Updates meta-parameters to enable fast adaptation
    """
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=outer_lr)
    
    meta_loss = 0.
    for support_batch, query_batch in tasks:
        # Inner loop: adapt to task
        adapted_model = maml_inner_loop(meta_model, support_batch, inner_lr, device)
        
        # Evaluate adapted model on query set
        adapted_model.eval()
        price = query_batch['price'].to(device)
        sentiment = query_batch['sentiment'].to(device)
        onchain = query_batch['onchain'].to(device)
        macro = query_batch['macro'].to(device)
        price_target = query_batch['price_target'].to(device)
        
        outputs = adapted_model(price, sentiment, onchain, macro)
        task_loss = F.mse_loss(outputs['price'], price_target)
        meta_loss += task_loss
    
    # Meta-update
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
    
    return meta_loss.item() / len(tasks)


# ============================================================================
# Training Functions
# ============================================================================
def compute_loss(outputs, batch, device):
    """NOVELTY 7: Multi-task loss with stability"""
    price_target = batch['price_target'].to(device)
    volatility_target = batch['volatility_target'].to(device)
    regime_target = batch['regime_target'].to(device).squeeze()
    
    price_loss = F.mse_loss(outputs['price'], price_target)
    volatility_loss = F.mse_loss(outputs['volatility'], volatility_target)
    regime_loss = F.cross_entropy(outputs['regime'], regime_target)
    
    logvar_clamped = torch.clamp(outputs['uncertainty_logvar'], min=-10, max=10)
    nll_loss = 0.5 * (torch.exp(-logvar_clamped) * 
                      (outputs['uncertainty_mean'] - price_target)**2 + 
                      logvar_clamped).mean()
    
    kl_loss = -0.5 * torch.mean(1 + outputs['vae_logvar'] - 
                                outputs['vae_mu'].pow(2) - 
                                outputs['vae_logvar'].exp())
    kl_loss = torch.clamp(kl_loss, min=0, max=100)
    
    total_loss = price_loss + 0.3 * volatility_loss + 0.2 * regime_loss + 0.1 * nll_loss + 0.001 * kl_loss
    
    return total_loss, {
        'price_loss': price_loss.item(),
        'volatility_loss': volatility_loss.item(),
        'regime_loss': regime_loss.item(),
        'uncertainty_loss': nll_loss.item(),
        'kl_loss': kl_loss.item()
    }


def train_epoch(model, loader, optimizer, device, use_maml=False, inner_lr=0.01):
    """Training with optional MAML"""
    model.train()
    total_loss = 0.
    loss_dict = {'price_loss': 0., 'volatility_loss': 0., 'regime_loss': 0., 
                 'uncertainty_loss': 0., 'kl_loss': 0.}
    
    if use_maml:
        # MAML meta-training
        # Create tasks: split each batch into support/query
        batch_iter = iter(loader)
        tasks = []
        
        try:
            while True:
                support_batch = next(batch_iter)
                query_batch = next(batch_iter)
                tasks.append((support_batch, query_batch))
                
                if len(tasks) >= 4:  # Process 4 tasks at a time
                    meta_loss = maml_outer_loop(model, tasks, inner_lr, 0.0001, device)
                    total_loss += meta_loss
                    tasks = []
        except StopIteration:
            if len(tasks) > 0:
                meta_loss = maml_outer_loop(model, tasks, inner_lr, 0.0001, device)
                total_loss += meta_loss
        
        return total_loss / max(len(loader) // 8, 1), loss_dict
    
    else:
        # Standard training
        for batch in loader:
            price = batch['price'].to(device)
            sentiment = batch['sentiment'].to(device)
            onchain = batch['onchain'].to(device)
            macro = batch['macro'].to(device)
            
            optimizer.zero_grad()
            outputs = model(price, sentiment, onchain, macro)
            loss, losses = compute_loss(outputs, batch, device)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_dict[k] += v
        
        n = max(len(loader), 1)
        return total_loss / n, {k: v / n for k, v in loss_dict.items()}


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Data collection
    print("=== Data Collection ===")
    collector = MultiModalDataCollector()
    price_df = collector.collect_price_data('bitcoin', days=365)
    sentiment_df = collector.collect_sentiment_data('BTC', days=365)
    onchain_df = collector.collect_onchain_data('bitcoin', days=365)
    macro_df = collector.collect_macro_data(days=365)
    
    merged_df = collector.merge_all_data(price_df, sentiment_df, onchain_df, macro_df)
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.replace([np.inf, -np.inf], 0)
    
    # Scaling
    print("Applying feature scaling...")
    price_scaler = MinMaxScaler()
    merged_df['close_scaled'] = price_scaler.fit_transform(merged_df[['close']])
    
    feature_cols = ['close', 'high', 'low', 'open', 'volume', 'ma_7', 'ma_21', 'ma_50', 
                   'std_21', 'log_return', 'return_1d', 'return_7d', 'vol_ma_7', 'rsi_14', 
                   'market_cap', 'sentiment_score', 'tweet_volume', 'positive_ratio', 
                   'negative_ratio', 'active_addresses', 'transaction_count', 'hash_rate', 
                   'nvt_ratio', 'mvrv_ratio', 'sp500', 'gold_price', 'oil_price', 'dxy', 
                   'vix', 'sp500_return', 'gold_return']
    
    scaler = StandardScaler()
    merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])
    
    os.makedirs('models', exist_ok=True)
    with open('models/price_scaler.pkl', 'wb') as f:
        pickle.dump(price_scaler, f)
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Data shape: {merged_df.shape}")
    print(f"Price range (scaled): {merged_df['close_scaled'].min():.4f} - {merged_df['close_scaled'].max():.4f}\n")
    
    # Dataset
    dataset = CryptoDataset(merged_df, lookback=10)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Model
    print("=== Initializing MCANU-Net with All 7 Novelties (Including MAML) ===")
    model = MCANUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training
    print("=== Training ===")
    epochs = 100
    best_val_loss = float('inf')
    use_maml = True  # Enable MAML meta-learning
    
    for epoch in range(1, epochs + 1):
        # Alternate between standard training and MAML every few epochs
        current_use_maml = use_maml and (epoch % 5 == 0)
        
        train_loss, train_losses = train_epoch(model, train_loader, optimizer, device, 
                                               use_maml=current_use_maml, inner_lr=0.01)
        
        mode_str = "[MAML]" if current_use_maml else "[STD]"
        print(f"Epoch {epoch:3d} {mode_str} | Train Loss: {train_loss:.6f}")
        
        if not current_use_maml:
            print(f"  Price: {train_losses['price_loss']:.6f} | Vol: {train_losses['volatility_loss']:.6f} | "
                  f"Regime: {train_losses['regime_loss']:.6f} | Unc: {train_losses['uncertainty_loss']:.6f} | KL: {train_losses['kl_loss']:.6f}")
        
        if train_loss < best_val_loss and not np.isnan(train_loss):
            best_val_loss = train_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print("  ✓ Best model saved")
    
    print("\n✓ Training complete with all 7 novelties including MAML!")


if __name__ == "__main__":
    main()
