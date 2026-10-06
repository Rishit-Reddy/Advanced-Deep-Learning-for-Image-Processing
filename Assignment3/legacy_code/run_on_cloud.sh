#!/bin/bash
# run_on_cloud.sh

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision optuna pandas pillow tqdm scikit-learn

# 2. Start the experiment runner in the background
echo "🚀 Starting Optuna Hyperparameter Sweep in background..."
nohup python3 experiment_runner.py > training_session.log 2>&1 &

echo "--------------------------------------------------------"
echo "✅ Training is now running in the background."
echo "📂 Logs: training_session.log"
echo "🔍 Monitor: tail -f training_session.log"
echo "--------------------------------------------------------"
