#!/usr/bin/env bash
set -e

mkdir -p configs data/raw data/processed src/{data,features,models,risk,backtest}
touch src/__init__.py src/data/__init__.py src/features/__init__.py src/models/__init__.py src/risk/__init__.py src/backtest/__init__.py

cat > requirements.txt << 'EOT'
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
PyYAML==6.0.2
EOT

cat > configs/default.yaml << 'EOT'
data:
  raw_prices_path: data/raw/sample_prices.csv
  processed_path: data/processed/features.csv
features:
  rsi_window: 14
  ma_fast: 10
  ma_slow: 30
model:
  target_horizon: 1
  test_size: 0.2
risk:
  max_weight_per_stock: 0.1
EOT

cat > src/main.py << 'EOT'
print("NSEStockAnalysis project scaffold ready.")
EOT

cat > data/raw/sample_prices.csv << 'EOT'
date,symbol,open,high,low,close,volume
2024-01-01,RELIANCE,2500,2520,2490,2510,1000000
2024-01-02,RELIANCE,2510,2530,2500,2525,1100000
EOT

echo "Scaffold created at: $(pwd)"
find . -maxdepth 3 -type f | sort
