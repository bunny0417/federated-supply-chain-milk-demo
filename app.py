import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
import time

# Import logic from main.py
from main import (
    load_model, 
    SCConfig, 
    POPULAR_CURRENCY_CODES,
    CURRENCY_NAMES,
    CURRENCY_SYMBOLS,
    build_round_context,
    build_insight_messages,
    build_chat_messages,
    generate_grounded_response,
    SupplyChainDataManager, 
    FedSim,
    LSTMModel, 
    optimize, 
    to_serializable,
    get_device
)

# Page Setup
st.set_page_config(
    page_title="Supply Chain 5.0 - FedSim",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

import matplotlib.pyplot as plt

def plot_financial_pie(financials, currency_code):
    labels = ['Projected Revenue', 'Order Cost', 'Potential Waste Cost']
    # Pie chart values: Revenue is positive, Costs are negative expenses but we plot magnitude
    # Actually, a better breakdown for parts-to-whole is: Profit + Cost + Waste = Total Revenue (if we sell all)
    # But Revenue = Profit + Cost + Waste is only true if we account for everything perfectly.
    # Let's plot: Cost vs Profit vs Waste
    
    sizes = [
        max(0, financials['net_profit']), 
        financials['order_cost'], 
        financials['waste_cost']
    ]
    labels = ['Net Profit', 'Order Cost', 'Waste Risk']
    colors = ['#4CAF50', '#FF9800', '#F44336'] # Green, Orange, Red
    explode = (0.1, 0, 0)  # explode the profit slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(f"Financial Mix ({currency_code})")
    return fig


def format_currency(amount, currency_code):
    symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
    return f"{symbol}{amount:.2f} {currency_code}"


def pick_directory(title, initial_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title=title, initialdir=initial_dir or os.getcwd())
        root.destroy()

        if chosen:
            return os.path.normpath(chosen)
    except Exception as e:
        st.sidebar.warning(f"Folder picker unavailable. Enter path manually. ({e})")
    return None

# Title
st.title("🤖 Federated Supply Chain Optimization")
st.markdown(f"**Device:** `{get_device()}` | **Model:** `{SCConfig.MODEL_NAME}`")

# Initialize sidebar-related session state before creating sidebar widgets
if "dataset_dir_text" not in st.session_state:
    st.session_state.dataset_dir_text = SCConfig.DATASET_DIR
if "log_dir_text" not in st.session_state:
    st.session_state.log_dir_text = SCConfig.LOG_DIR
if "manual_currency_code" not in st.session_state:
    st.session_state.manual_currency_code = SCConfig.CURRENCY
if "effective_currency" not in st.session_state:
    st.session_state.effective_currency = SCConfig.CURRENCY
if "currency_mode" not in st.session_state:
    st.session_state.currency_mode = "manual"
if "mixed_currency_warning" not in st.session_state:
    st.session_state.mixed_currency_warning = False
if "currency_detection_details" not in st.session_state:
    st.session_state.currency_detection_details = {}

# Sidebar - Configuration
st.sidebar.header("Configuration")
st.sidebar.info(f"**Product:** {SCConfig.PRODUCT_NAME}") # Display as info, fixed
num_clients = st.sidebar.slider("Number of Clients", 1, 10, SCConfig.NUM_CLIENTS)
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, SCConfig.NUM_ROUNDS)
carbon_cap = st.sidebar.number_input("Carbon Cap", value=SCConfig.CARBON_CAP)
dp_epsilon = st.sidebar.slider("DP Epsilon (Privacy Budget)", 0.1, 20.0, SCConfig.DP_EPSILON, help="Lower = More Noise/Privacy")
explanation_max_tokens = st.sidebar.slider("Explanation Max Tokens", 60, 260, 120, step=20)
data_source_mode = st.sidebar.selectbox("Data Source", ["real", "synthetic"], index=0 if SCConfig.DATA_SOURCE_MODE == "real" else 1)
manual_currency_index = 0
if st.session_state.manual_currency_code in POPULAR_CURRENCY_CODES:
    manual_currency_index = POPULAR_CURRENCY_CODES.index(st.session_state.manual_currency_code)

manual_currency = st.sidebar.selectbox(
    "Manual Currency (Fallback)",
    POPULAR_CURRENCY_CODES,
    index=manual_currency_index,
    key="manual_currency_code",
    format_func=lambda code: f"{code} - {CURRENCY_NAMES.get(code, code)} ({CURRENCY_SYMBOLS.get(code, code)})"
)

st.sidebar.text_input("Dataset Directory", key="dataset_dir_text")
if st.sidebar.button("Browse Dataset Directory"):
    selected = pick_directory("Select Dataset Directory", st.session_state.dataset_dir_text)
    if selected:
        st.session_state.dataset_dir_text = selected
        st.rerun()

st.sidebar.text_input("Log Directory", key="log_dir_text")
if st.sidebar.button("Browse Log Directory"):
    selected = pick_directory("Select Log Directory", st.session_state.log_dir_text)
    if selected:
        st.session_state.log_dir_text = selected
        st.rerun()


# Update Config
SCConfig.NUM_CLIENTS = num_clients
SCConfig.NUM_ROUNDS = num_rounds
SCConfig.CARBON_CAP = carbon_cap
SCConfig.DP_EPSILON = dp_epsilon
SCConfig.DATA_SOURCE_MODE = data_source_mode
SCConfig.CURRENCY = manual_currency

dataset_dir = st.session_state.dataset_dir_text.strip()
log_dir = st.session_state.log_dir_text.strip()
SCConfig.DATASET_DIR = os.path.normpath(dataset_dir) if dataset_dir else ""
SCConfig.LOG_DIR = os.path.normpath(log_dir) if log_dir else ""

if st.session_state.currency_mode != "auto-detected":
    st.session_state.effective_currency = SCConfig.CURRENCY

effective_currency = st.session_state.get("effective_currency", SCConfig.CURRENCY)
effective_symbol = CURRENCY_SYMBOLS.get(effective_currency, effective_currency)
if st.session_state.currency_mode == "auto-detected":
    st.sidebar.success(f"Currency in use: {effective_currency} ({effective_symbol}) [auto-detected]")
else:
    st.sidebar.info(f"Currency in use: {effective_currency} ({effective_symbol}) [manual fallback]")

if st.session_state.mixed_currency_warning:
    st.sidebar.warning("Mixed currencies detected across files. Using the first detected currency.")

if st.session_state.currency_mode == "auto-detected" and st.session_state.currency_detection_details:
    first_file = next(iter(st.session_state.currency_detection_details.keys()))
    first_detail = st.session_state.currency_detection_details[first_file]
    st.sidebar.caption(
        f"Detection source: {first_file} -> {first_detail.get('currency')} ({first_detail.get('source')})"
    )

if SCConfig.DATA_SOURCE_MODE == "real" and SCConfig.DATASET_DIR and not os.path.isdir(SCConfig.DATASET_DIR):
    st.sidebar.warning(f"Dataset directory not found: {SCConfig.DATASET_DIR}")

if SCConfig.LOG_DIR:
    try:
        os.makedirs(SCConfig.LOG_DIR, exist_ok=True)
    except OSError as e:
        st.sidebar.error(f"Log directory is not writable: {e}")

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_manager" not in st.session_state:
    st.session_state.data_manager = None
if "client0_max" not in st.session_state:
    st.session_state.client0_max = None
if "round_context" not in st.session_state:
    st.session_state.round_context = None

# Main Layout: 2 Columns
# split into [Left: Simulation/Results, Right: Chat Assistant]
main_col, chat_col = st.columns([7, 3])

with main_col:
    # Title
    st.title("🤖 Federated Supply Chain Optimization")
    st.markdown(f"**Device:** `{get_device()}` | **Model:** `{SCConfig.MODEL_NAME}`")

    # Model Loading Section
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Load Model"):
            with st.spinner("Loading Model... This may take a while."):
                try:
                    tokenizer, model = load_model()
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.success("Model Loaded!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
    with col2:
        if st.session_state.model:
            st.info("✅ Model Active")
        else:
            st.warning("⚠️ Model not loaded")

    # Simulation Section
    if st.session_state.model:
        st.divider()
        st.header("Simulation Control")
        
        # We use a session state flag to trigger run vs just button
        if st.button("Run Federated Simulation"):
            st.session_state.run_simulation = True
            
        if st.session_state.get("run_simulation", False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Initialize Data (Only if not already active? No, re-run means fresh data/round)
            status_text.text(f"Loading {SCConfig.DATA_SOURCE_MODE} client data...")
            data_manager = SupplyChainDataManager(
                SCConfig.NUM_CLIENTS,
                data_mode=SCConfig.DATA_SOURCE_MODE,
                dataset_dir=SCConfig.DATASET_DIR,
                manual_currency=SCConfig.CURRENCY
            )
            st.session_state.data_manager = data_manager
            st.session_state.effective_currency = data_manager.get_effective_currency()
            st.session_state.currency_mode = "auto-detected" if data_manager.is_currency_auto_detected() else "manual"
            st.session_state.mixed_currency_warning = data_manager.has_mixed_currency()
            st.session_state.currency_detection_details = data_manager.currency_detection_details
            progress_bar.progress(20)
            
            # 2. Run FedSim
            status_text.text(f"Running Federated Learning for {SCConfig.NUM_ROUNDS} rounds...")
            fed = FedSim(data_manager)
            
            # Run with DP Epsilon
            lstm_model, metrics = fed.run(st.session_state.tokenizer, st.session_state.model, epsilon=SCConfig.DP_EPSILON)
            progress_bar.progress(80)
            
            # 3. Optimization using LSTM Forecast
            status_text.text("Optimizing for Client 0...")
            client0_df = data_manager.get_client_data("0")
            
            # Prepare input for LSTM (Last 5 weeks)
            data = client0_df["demand"].values.astype(np.float32)
            max_val = fed.max_vals.get("0", float(np.max(data)) if np.max(data) > 0 else 1.0)
            st.session_state.client0_max = max_val
            last_seq = data[-5:] / max_val
            inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
            
            lstm_model.eval()
            with torch.no_grad():
                pred_norm = lstm_model(inp).item()
            
            forecast = int(pred_norm * max_val)
            
            # Get Context Data
            client0 = client0_df.iloc[-1]
            
            opt = optimize(
                forecast=forecast,
                inventory=50,
                emission_factor=float(client0["emission_factor"]),
                risk=float(client0["disruption_prob"])
            )
            
            # Store constraints context for Chat
            st.session_state.forecast = forecast
            st.session_state.opt_result = opt
            st.session_state.metrics = metrics
            st.session_state.simulation_done = True
            st.session_state.round_context = build_round_context(
                opt,
                forecast,
                st.session_state.effective_currency,
                disruption_prob=float(client0["disruption_prob"]),
                emission_factor=float(client0["emission_factor"]),
            )
            
            # We DON'T clear messages on re-run so user keeps chat history across rounds
            # st.session_state.messages = [] 
            
            progress_bar.progress(100)
            status_text.text("Simulation Complete!")
            
            # Turn off trigger to prevent infinite loop if we weren't careful 
            # (though in Streamlit button click is ephemeral, this persists result display)
            st.session_state.run_simulation = False

    # Results Section
    if st.session_state.simulation_done and st.session_state.opt_result:
        st.divider()
        st.header("Results & Recommendation")
        data_manager = st.session_state.get("data_manager")
        
        opt = st.session_state.opt_result
        forecast = st.session_state.forecast
        metrics = st.session_state.metrics
        effective_currency = st.session_state.get("effective_currency", SCConfig.CURRENCY)
        
        # Training Metrics
        st.subheader("Training Performance")
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric("Final MAE", f"{metrics['mae'][-1]:.2f}")
        with met_col2:
            st.metric("Final RMSE", f"{metrics['rmse'][-1]:.2f}")
            
        # Charts
        st.caption("Federated Training Loss (MSE)")
        chart_data = pd.DataFrame({
            "Round": metrics["rounds"],
            "Training Loss": metrics["loss"],
            "MAE": metrics["mae"]
        })
        st.line_chart(chart_data, x="Round", y=["Training Loss", "MAE"])
        
        st.divider()

        # Historical Trend Visualization
        st.subheader("Historical Demand & Forecast")
        if data_manager is None:
            data_manager = SupplyChainDataManager(
                SCConfig.NUM_CLIENTS,
                data_mode=SCConfig.DATA_SOURCE_MODE,
                dataset_dir=SCConfig.DATASET_DIR,
                manual_currency=SCConfig.CURRENCY
            )
            st.session_state.data_manager = data_manager
            st.session_state.effective_currency = data_manager.get_effective_currency()
            st.session_state.currency_mode = "auto-detected" if data_manager.is_currency_auto_detected() else "manual"
            st.session_state.mixed_currency_warning = data_manager.has_mixed_currency()
            st.session_state.currency_detection_details = data_manager.currency_detection_details
            effective_currency = st.session_state.get("effective_currency", SCConfig.CURRENCY)

        client0_df = data_manager.get_client_data("0")
        
        # Get last 20 weeks for better visibility (or full history)
        history_df = client0_df.tail(20).copy()
        history_df["Type"] = "Historical"
        
        # Create a tiny dataframe for the forecast point
        last_week = history_df.index[-1]
        next_week_idx = last_week + 1 # Simple index increment
        
        # We need to structure it for the chart
        # Let's use a simple line chart with the forecast appended
        
        if "date" in history_df.columns and history_df["date"].notna().any():
            chart_data = history_df[["date", "demand"]].set_index("date")
        else:
            chart_data = history_df[["week", "demand"]].set_index("week")
        
        # Add Forecast row
        # We can't easily mix types in simple st.line_chart, so we plot history 
        # and maybe add a distinct marker or just append it.
        # Let's append it as a continuation
        
        st.line_chart(chart_data)
        st.caption(f"Demand data for the last 20 weeks. Forecast for next week: **{forecast}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecast Demand", f"{forecast} units")
        m2.metric("Recommended Order", f"{opt['optimized_qty']} units")
        m3.metric("Projected Emissions", f"{opt['emissions']:.2f}")
        m4.metric("Feasible?", "✅ Yes" if opt['feasible'] else "❌ No")

        # Financial Pie Chart
        st.subheader("Financial Projection")
        fin = opt['financials']
        
        f1, f2 = st.columns([1, 2])
        with f1:
            st.metric("Proj. Revenue", format_currency(fin['revenue'], effective_currency))
            st.metric("Net Profit", format_currency(fin['net_profit'], effective_currency), delta_color="normal")
        with f2:
            st.write(f"Current round financial composition ({effective_currency}):")
            st.pyplot(plot_financial_pie(fin, effective_currency))

        
        # LLM Recommendation
        st.subheader("AI Insight")
        if st.button("Generate Explanation"):
            with st.spinner("Asking TinyLlama..."):
                latest_row = client0_df.iloc[-1]
                insight_context = build_round_context(
                    opt,
                    forecast,
                    effective_currency,
                    disruption_prob=float(latest_row["disruption_prob"]),
                    emission_factor=float(latest_row["emission_factor"]),
                )
                st.session_state.round_context = insight_context
                insight_messages = build_insight_messages(insight_context)
                insight_result = generate_grounded_response(
                    insight_messages,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    max_tokens=explanation_max_tokens,
                    mode="insight",
                    context=insight_context,
                    retry_on_failure=False,
                )
                st.info(insight_result["text"])
                if insight_result.get("used_fallback"):
                    st.caption("Low-confidence model output detected. A deterministic tactical fallback was shown.")
                
        # Manual Override
        st.divider()
        st.subheader("Decision & Next Round")
        
        with st.form("override_form"):
            new_qty = st.number_input("Adjust Order Quantity", value=int(opt['optimized_qty']))
            submitted = st.form_submit_button("Approve & Run Next Round")
            
            if submitted:
                if new_qty != opt['optimized_qty']:
                    log_entry = {"event": "override", "new": int(new_qty), "original": opt['optimized_qty'], "product": SCConfig.PRODUCT_NAME}
                    st.warning(f"Order quantity overridden to {new_qty}")
                else:
                    log_entry = {"event": "approved", "qty": opt['optimized_qty'], "product": SCConfig.PRODUCT_NAME}
                    st.success("AI Recommendation Approved")
                
                # Save Log
                os.makedirs(SCConfig.LOG_DIR, exist_ok=True)
                with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "a") as f:
                    f.write(json.dumps(to_serializable(log_entry)) + "\n")
                st.toast("Decision saved! Starting next round...")
                
                # Trigger Re-run
                time.sleep(1) # Visual pause
                st.session_state.run_simulation = True
                st.rerun()


# --- Chat Interface (Right Column) ---
with chat_col:
    st.header("💬 Chat Assistant")
    st.caption(f"Topic: {SCConfig.PRODUCT_NAME}")
    
    # Initialize chat container to keep it scrollable/separate
    chat_container = st.container(height=600)
    
    with chat_container:
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat Input
        if prompt := st.chat_input(f"Ask about {SCConfig.PRODUCT_NAME}..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            chat_context = st.session_state.get("round_context")
            if chat_context is None and st.session_state.simulation_done and st.session_state.opt_result:
                dm = st.session_state.get("data_manager")
                if dm is not None:
                    last_row = dm.get_client_data("0").iloc[-1]
                    chat_context = build_round_context(
                        st.session_state.opt_result,
                        st.session_state.forecast,
                        st.session_state.get("effective_currency", SCConfig.CURRENCY),
                        disruption_prob=float(last_row["disruption_prob"]),
                        emission_factor=float(last_row["emission_factor"]),
                    )
                    st.session_state.round_context = chat_context
            messages = build_chat_messages(prompt, chat_context)
            
            # Generate Response
            if st.session_state.model:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_result = generate_grounded_response(
                            messages,
                            st.session_state.tokenizer,
                            st.session_state.model,
                            max_tokens=220,
                            mode="chat",
                            context=chat_context,
                            retry_on_failure=True,
                        )
                        response = response_result["text"]
                        st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Please load the model first.")

