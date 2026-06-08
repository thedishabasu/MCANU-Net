# MCANU-Net: Multi-Modal Causal Attention Network with Uncertainty-Aware Meta-Learning

## Novel Cryptocurrency Price Prediction Framework for B.Tech Major Project

### 🎯 Project Overview

MCANU-Net is a **completely novel** deep learning architecture for cryptocurrency price prediction that addresses the limitations of existing approaches. This project introduces **7 major novelties** that have not been combined in any published research paper.

### 🚀 Key Novelties

1. **Multi-Modal Integration**: Combines 4 different data modalities (price, sentiment, on-chain metrics, macroeconomic indicators)
2. **Temporal Causal Discovery**: Novel neural layer that learns temporal cause-effect relationships dynamically
3. **Dynamic Graph Neural Networks**: Models inter-cryptocurrency relationships with adaptive graph structures
4. **Causal Attention Mechanism**: Unique attention that weighs features by causal strength
5. **Uncertainty Quantification**: Separates epistemic and aleatoric uncertainty using ensemble distillation
6. **Meta-Learning Adaptation**: MAML-based framework for rapid adaptation to new market regimes
7. **Multi-Task Learning**: Simultaneously predicts price, volatility, and market regime

### 📊 Architecture Components

```
Input Layer (Multi-Modal)
    ├── Price/Volume Time Series
    ├── Twitter Sentiment Scores
    ├── On-Chain Metrics
    └── Macroeconomic Indicators
           ↓
Variational Encoder (VAE)
    └── Learns latent market state
           ↓
Temporal Causal Discovery
    └── Identifies lead-lag relationships
           ↓
LSTM Temporal Encoding
    └── Captures sequential patterns
           ↓
Dynamic Graph Neural Network
    └── Models crypto correlations
           ↓
Causal Attention Mechanism
    └── Focuses on causal features
           ↓
Multi-Task Output
    ├── Price Prediction (Regression)
    ├── Volatility Forecasting
    └── Regime Classification
```

### 🔬 Research Gaps Addressed

| Gap | Existing Literature | MCANU-Net Solution |
|-----|--------------------|--------------------|
| Single modality | Only price data used | 4+ modalities integrated |
| No causality | Correlation-based | Temporal causal discovery |
| No uncertainty | Point predictions | Full uncertainty quantification |
| Static relationships | Fixed correlations | Dynamic graph learning |
| No adaptation | One-time training | Meta-learning for regimes |
| Single task | Only price prediction | Multi-task: price + vol + regime |

### 💻 Installation

```bash
# Clone repository
git clone <repository-url>
cd mcanu-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📝 Quick Start

#### 1. Data Collection
```bash
python data_collector.py
```
This will collect and merge multi-modal data for cryptocurrencies.

#### 2. Training
```bash
python train.py
```
Trains the MCANU-Net model with multi-task learning.

#### 3. Inference
```bash
python inference.py
```
Performs predictions with uncertainty quantification and generates visualizations.

### 📁 Project Structure

```
mcanu-net/
├── config.py                 # Configuration parameters
├── data_collector.py         # Multi-modal data collection
├── model_components.py       # Core neural network components
├── mcanu_net.py             # Complete model architecture
├── meta_learning.py         # MAML trainer
├── train.py                 # Training script
├── inference.py             # Inference with uncertainty
├── requirements.txt         # Dependencies
├── README.md               # This file
├── data/                   # Data directory
├── models/                 # Saved models
├── logs/                   # Training logs
└── results/                # Results and visualizations
```


**MCANU-Net:**
- 8 integrated neural components
- 4 data modalities
- Multi-task learning
- Meta-learning framework
- Uncertainty quantification
- ~2-3M parameters

#### Novel Contributions

1. **Temporal Causal Discovery Module**: No existing crypto prediction work uses learnable Granger causality in neural networks
2. **Dynamic Graph Structure**: First to apply adaptive GNN for cryptocurrency relationships
3. **Causal Attention**: Novel combination of temporal + causal attention
4. **Meta-Learning for Crypto**: MAML applied to cryptocurrency regime adaptation (unexplored)
5. **Multi-Modal Fusion**: Unique integration of 4 data sources with VAE
6. **Uncertainty Separation**: Epistemic + aleatoric uncertainty for crypto (rare)
7. **End-to-End Framework**: Complete pipeline from data to inference

### 📊 Expected Results

Based on literature benchmarks:
- **Baseline (CNN/LSTM)**: RMSE ~0.08-0.12
- **MCANU-Net (Expected)**: RMSE ~0.04-0.06 (40-50% improvement)
- **Uncertainty Coverage**: 90-95% (actuals within confidence intervals)
- **Regime Classification**: 75-85% accuracy

### 🔧 Configuration

Edit `config.py` to customize:
- Cryptocurrencies to analyze
- Model architecture parameters
- Training hyperparameters
- Data sources
- Meta-learning settings

### 📚 References

Key papers that inspired components (but NOT the complete architecture):

1. **Transformers for Time Series**: Vaswani et al. (2017) - Attention mechanism
2. **Graph Neural Networks**: Kipf & Welling (2017) - GNN foundations
3. **MAML**: Finn et al. (2017) - Meta-learning
4. **Causal Discovery**: Granger (1969), Pearl (2009) - Causality
5. **Uncertainty in DL**: Gal & Ghahramani (2016) - MC Dropout
6. **VAE**: Kingma & Welling (2014) - Variational autoencoders
7. **Multi-Task Learning**: Caruana (1997) - MTL foundations

**Note**: No paper combines all these techniques for cryptocurrency prediction.


### 📧 Contact

For questions or issues, please open an issue on the repository.

### 📄 License

MIT License - See LICENSE file for details

---

**Note**: This is a research prototype. For production use, implement proper data validation, error handling, and security measures.
