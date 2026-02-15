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
    SupplyChainDataManager, 
    FedSim,
    LSTMModel, 
    optimize, 
    llm_generate, 
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

def plot_financial_pie(financials):
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
    return fig

# Title
st.title("🤖 Federated Supply Chain Optimization")
st.markdown(f"**Device:** `{get_device()}` | **Model:** `{SCConfig.MODEL_NAME}`")

# Sidebar - Configuration
st.sidebar.header("Configuration")
st.sidebar.info(f"**Product:** {SCConfig.PRODUCT_NAME}") # Display as info, fixed
num_clients = st.sidebar.slider("Number of Clients", 1, 10, SCConfig.NUM_CLIENTS)
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, SCConfig.NUM_ROUNDS)
carbon_cap = st.sidebar.number_input("Carbon Cap", value=SCConfig.CARBON_CAP)
dp_epsilon = st.sidebar.slider("DP Epsilon (Privacy Budget)", 0.1, 20.0, SCConfig.DP_EPSILON, help="Lower = More Noise/Privacy")
log_dir = st.sidebar.text_input("Log Directory", SCConfig.LOG_DIR)


# Update Config
SCConfig.NUM_CLIENTS = num_clients
SCConfig.NUM_ROUNDS = num_rounds
SCConfig.CARBON_CAP = carbon_cap
SCConfig.LOG_DIR = log_dir
SCConfig.DP_EPSILON = dp_epsilon

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
            status_text.text("Generating synthetic data...")
            data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)
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
            max_val = 300.0
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
        
        opt = st.session_state.opt_result
        forecast = st.session_state.forecast
        metrics = st.session_state.metrics
        
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
        client0_df = data_manager.get_client_data("0")
        
        # Get last 20 weeks for better visibility (or full history)
        history_df = client0_df.tail(20).copy()
        history_df["Type"] = "Historical"
        
        # Create a tiny dataframe for the forecast point
        last_week = history_df.index[-1]
        next_week_idx = last_week + 1 # Simple index increment
        
        # We need to structure it for the chart
        # Let's use a simple line chart with the forecast appended
        
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
            st.metric("Proj. Revenue", f"${fin['revenue']:.2f}")
            st.metric("Net Profit", f"${fin['net_profit']:.2f}", delta_color="normal")
        with f2:
            st.write("Current round financial composition:")
            st.pyplot(plot_financial_pie(fin))

        
        # LLM Recommendation
        st.subheader("AI Insight")
        if st.button("Generate Explanation"):
            with st.spinner("Asking TinyLlama..."):
                system_msg = {
                    "role": "system", 
                    "content": f"You are a supply chain expert. Product: {SCConfig.PRODUCT_NAME}. Forecast: {forecast}. Emissions: {opt['emissions']:.2f}."
                }
                user_msg = {
                    "role": "user", 
                    "content": f"The recommended order quantity is {opt['optimized_qty']}. Provide a brief strategic recommendation."
                }
                
                insight = llm_generate(
                    [system_msg, user_msg],
                    st.session_state.tokenizer,
                    st.session_state.model,
                    max_tokens=150
                )
                st.info(insight)
                
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
                
            # Construct Context from session state if available
            context_str = ""
            if st.session_state.simulation_done and st.session_state.opt_result:
                opt = st.session_state.opt_result
                forecast = st.session_state.forecast
                context_str = (
                    f"Context: Managing supply chain for '{SCConfig.PRODUCT_NAME}'. "
                    f"Forecast: {forecast} units. "
                    f"Recommended Order: {opt['optimized_qty']} units. "
                    f"Emissions: {opt['emissions']:.2f}. "
                )
            
            messages = [
                {"role": "system", "content": f"You are a helpful Supply Chain Assistant. {context_str}"},
                {"role": "user", "content": prompt}
            ]
            
            # Generate Response
            if st.session_state.model:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = llm_generate(
                            messages, 
                            st.session_state.tokenizer,
                            st.session_state.model,
                            max_tokens=200
                        )
                        st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Please load the model first.")

