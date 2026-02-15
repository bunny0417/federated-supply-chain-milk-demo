"""
Supply Chain 5.0
Local VS Code Version
Real AI using TinyLlama 1.1B (4-bit)
Federated Simulation + Optimization + Human Override + AI Impact
"""

import os
import gc
import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =====================================================
# Configuration
# =====================================================
@dataclass
class SCConfig:
    MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PRODUCT_NAME: str = "Milk"  # Fixed Product
    NUM_CLIENTS: int = 3
    NUM_ROUNDS: int = 2
    CARBON_CAP: float = 500.0  # Adjusted for Milk (e.g. per batch)
    LOG_DIR: str = "sc50_logs"
    DP_EPSILON: float = 5.0  # Privacy Budget (Lower = More Privacy/Noise)
    
    # Financials (Per Unit)
    SELLING_PRICE: float = 4.0
    COST_PRICE: float = 1.5
    WASTE_COST: float = 0.5  # Cost of disposal/spoilage


# Create logs folder if missing
os.makedirs(SCConfig.LOG_DIR, exist_ok=True)


# =====================================================
# Utility
# =====================================================
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj


# =====================================================
# Load TinyLlama
# =====================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model():
    device = get_device()
    log(f"Loading TinyLlama 1.1B on {device}...")

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            SCConfig.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU or MPS (Apple Silicon) - 4-bit quantization usually requires CUDA
        # We load in float32 (default) or float16 if supported to save memory
        torch_dtype = torch.float32 
        if device == "mps":
             torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            SCConfig.MODEL_NAME,
            device_map=device,
            torch_dtype=torch_dtype
        )

    tokenizer = AutoTokenizer.from_pretrained(SCConfig.MODEL_NAME)
    
    log("Model Loaded Successfully.")
    return tokenizer, model


def llm_generate(prompt, tokenizer, model, max_tokens=200, temperature=0.7):
    # Support both raw string prompts and chat messages list
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": f"You are a helpful Supply Chain Assistant optimized for {SCConfig.PRODUCT_NAME}."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt

    # Apply Chat Template (handles <|system|>, <|user|>, etc. for TinyLlama)
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,      
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1 # Prevent repetition
        )

    # Decode only the new tokens (response)
    # outputs contains [input_ids + new_tokens]
    response_ids = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# =====================================================
# Differential Privacy
# =====================================================
class DifferentialPrivacy:
    @staticmethod
    def add_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
        """Adds Laplacian noise for Differential Privacy."""
        if epsilon <= 0: return value # No privacy
        beta = sensitivity / epsilon
        noise = np.random.laplace(0, beta)
        return value + noise

    @staticmethod
    def clip_gradients(value: float, clip_norm: float = 5.0) -> float:
        """Clips the update to bound sensitivity."""
        return max(min(value, clip_norm), -clip_norm)


# =====================================================
# Synthetic Data
# =====================================================
class SupplyChainDataManager:
    def __init__(self, num_clients: int, weeks: int = 52):
        self.num_clients = num_clients
        self.weeks = weeks
        self.client_data = {}
        self.generate_data()

    def generate_data(self):
        np.random.seed(42)
        for cid in range(self.num_clients):
            t = np.arange(self.weeks)
            trend = 100 + (t * 0.5)
            seasonality = 20 * np.sin(2 * np.pi * t / 12)
            noise = np.random.normal(0, 5, self.weeks)
            firm_shift = np.random.randint(-10, 20)

            demand = trend + seasonality + noise + firm_shift
            disruption_prob = np.clip(np.random.beta(2, 10, self.weeks), 0, 1)
            emission_factor = np.full(self.weeks, 1.5 + (np.random.rand() * 0.5))

            self.client_data[str(cid)] = pd.DataFrame({
                "week": t,
                "demand": demand.astype(int),
                "disruption_prob": disruption_prob,
                "emission_factor": emission_factor
            })

    def get_client_data(self, cid: str):
        return self.client_data[str(cid)]


# =====================================================
# Federated LSTM Model
# =====================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def federated_average(models_state_dict):
    """Averages the weights of multiple models."""
    global_dict = models_state_dict[0].copy()
    for k in global_dict.keys():
        for i in range(1, len(models_state_dict)):
            global_dict[k] += models_state_dict[i][k]
        global_dict[k] = torch.div(global_dict[k], len(models_state_dict))
    return global_dict


# =====================================================
# Federated Simulation (LSTM)
# =====================================================
class FedSim:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.input_size = 1
        self.sequence_length = 5
        # Initialize Global Model
        self.global_model = LSTMModel(input_size=self.input_size)
        self.metrics = {"rounds": [], "mae": [], "rmse": [], "loss": []}

    def train_client(self, cid, global_weights, epochs=5, lr=0.01):
        """Trains a local model on client data."""
        # Load local model with global weights
        local_model = LSTMModel(input_size=self.input_size)
        local_model.load_state_dict(global_weights)
        local_model.train()
        
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare Data
        df = self.data_manager.get_client_data(str(cid))
        data = df["demand"].values.astype(np.float32)
        
        # Normalize Data (Simple MinMax for stability, ideally learned globally but approximating here)
        self.max_val = 300.0 # Approximate max demand
        data_norm = data / self.max_val

        # Create Sequences
        X, y = [], []
        for i in range(len(data_norm) - self.sequence_length):
            X.append(data_norm[i:i+self.sequence_length])
            y.append(data_norm[i+self.sequence_length])
            
        X = torch.tensor(X).unsqueeze(-1) # (Batch, Seq, Feature)
        y = torch.tensor(y).unsqueeze(-1) # (Batch, 1)
        
        # Local Training Loop
        epoch_loss = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        return local_model.state_dict(), epoch_loss / epochs

    def run(self, tokenizer=None, model=None, epsilon=SCConfig.DP_EPSILON):
        # NOTE: tokenizer/model args kept for compatibility but not used for LSTM training
        log("Starting Federated LSTM Simulation")
        
        self.metrics = {"rounds": [], "mae": [], "rmse": [], "loss": []}
        
        for r in range(SCConfig.NUM_ROUNDS):
            log(f"--- Round {r+1} ---")
            local_weights = []
            round_loss = 0
            
            # Broadcast Global Weights
            global_weights = self.global_model.state_dict()
            
            for cid in range(SCConfig.NUM_CLIENTS):
                # Train Client
                w, loss = self.train_client(cid, global_weights)
                
                # --- Differential Privacy (Add Noise to Weights) ---
                # Simple implementation: Add noise to each weight tensor
                if epsilon > 0:
                    for k in w.keys():
                        noise = torch.tensor(np.random.laplace(0, 0.01 / epsilon, w[k].shape)).float()
                        w[k] += noise
                # ---------------------------------------------------
                
                local_weights.append(w)
                round_loss += loss
                
            # Aggregation (FedAvg)
            new_global_weights = federated_average(local_weights)
            self.global_model.load_state_dict(new_global_weights)
            
            # Validation (Metrics on all clients)
            # Use the new global model to predict last known data point
            total_mae = 0
            total_rmse = 0
            
            self.global_model.eval()
            with torch.no_grad():
                for cid in range(SCConfig.NUM_CLIENTS):
                    df = self.data_manager.get_client_data(str(cid))
                    data = df["demand"].values.astype(np.float32)
                    
                    # Predict last week using previous sequence
                    last_seq = data[-self.sequence_length-1:-1] / self.max_val
                    true_val = data[-1]
                    
                    inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
                    pred_norm = self.global_model(inp).item()
                    pred = int(pred_norm * self.max_val)
                    
                    err = abs(true_val - pred)
                    total_mae += err
                    total_rmse += err**2
                    
            avg_loss = round_loss / SCConfig.NUM_CLIENTS
            mae = total_mae / SCConfig.NUM_CLIENTS
            rmse = np.sqrt(total_rmse / SCConfig.NUM_CLIENTS)
            
            self.metrics["rounds"].append(r+1)
            self.metrics["mae"].append(mae)
            self.metrics["rmse"].append(rmse)
            self.metrics["loss"].append(avg_loss)

            log(f"Round {r+1} | Loss: {avg_loss:.4f} | MAE: {mae:.2f}")

        return self.global_model, self.metrics


# =====================================================
# Optimization
# =====================================================
def optimize(forecast, inventory, emission_factor, risk):
    # Safety Stock includes risk buffer
    safety_stock = int(forecast * (0.1 + risk))
    
    # Order Qty logic
    qty = max(0, forecast + safety_stock - inventory)

    # Emissions
    emissions = float(qty * emission_factor)
    feasible = emissions <= SCConfig.CARBON_CAP
    
    # Financials (Projected)
    # Scenario: We sell everything we forecast (up to available stock)
    # Available for sale = Inventory + Qty
    available_stock = inventory + qty
    projected_sales = min(forecast, available_stock)
    unsold_stock = max(0, available_stock - projected_sales)
    
    revenue = projected_sales * SCConfig.SELLING_PRICE
    cost = qty * SCConfig.COST_PRICE # Cost of new order
    # Note: Logic for "Profit" usually includes Cost of Goods Sold (COGS). 
    # Here we simplify: Project Cost = Cost of New Order + Holding/Waste of Unsold.
    
    # Assuming unsold milk spoils (Waste Cost)
    waste_cost = unsold_stock * SCConfig.WASTE_COST
    
    net_profit = revenue - cost - waste_cost

    return {
        "optimized_qty": qty,
        "emissions": emissions,
        "feasible": feasible,
        "safety_stock": safety_stock,
        "financials": {
            "revenue": revenue,
            "order_cost": cost,
            "waste_cost": waste_cost,
            "net_profit": net_profit
        }
    }


# =====================================================
# Main Pipeline
# =====================================================
def main():
    device = get_device()
    log(f"Running on {device}")

    # Load LLM for Explanation only
    tokenizer, model = load_model()

    data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)

    # Federated LSTM Training
    fed = FedSim(data_manager)
    lstm_model, metrics = fed.run(tokenizer, model)

    # FINAL FORECAST (Using trained LSTM)
    client0_df = data_manager.get_client_data("0")
    data = client0_df["demand"].values.astype(np.float32)
    max_val = 300.0 # Same normalization constant as in training
    
    # Get last 5 weeks
    last_seq = data[-5:] / max_val
    inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
    
    lstm_model.eval()
    with torch.no_grad():
        pred_norm = lstm_model(inp).item()
        
    forecast = int(pred_norm * max_val)
    
    # Get last known emission/risk factors
    last_week_data = client0_df.iloc[-1]
    
    log(f"Final LSTM Forecast: {forecast}")

    opt = optimize(
        forecast=forecast,
        inventory=50,
        emission_factor=float(last_week_data["emission_factor"]),
        risk=float(last_week_data["disruption_prob"])
    )

    print("\nAI Recommendation (Explanation):\n")
    print(llm_generate(
        f"Forecast: {forecast}, Order Qty: {opt['optimized_qty']}, Emissions: {opt['emissions']}. Provide recommendation.",
        tokenizer,
        model,
        max_tokens=120,
        temperature=0.7 
    ))

    print("\nSuggested Order:", opt["optimized_qty"])
    print(f"Final MAE: {metrics['mae'][-1]:.2f}")
    
    user = input("Press Enter to approve or type new quantity: ").strip()

    if user.isdigit():
        new_qty = int(user)
        log_entry = {"event": "override", "new": new_qty}
    else:
        log_entry = {"event": "approved", "qty": opt['optimized_qty']}

    with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "w") as f:
        json.dump(to_serializable(log_entry), f, indent=2)

    log("Decision saved.")


if __name__ == "__main__":
    main()
