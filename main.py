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
from typing import Any, Dict, List, Optional

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
    DATA_SOURCE_MODE: str = "real"  # "real" or "synthetic"
    DATASET_DIR: str = "DATASETS"
    CURRENCY: str = "INR"  # Manual fallback currency (used when auto-detection fails)
    
    # Financials (Per Unit)
    SELLING_PRICE: float = 4.0
    COST_PRICE: float = 1.5
    WASTE_COST: float = 0.5  # Cost of disposal/spoilage


# Create logs folder if missing
os.makedirs(SCConfig.LOG_DIR, exist_ok=True)


POPULAR_CURRENCY_CODES = [
    "INR", "USD", "EUR", "GBP", "JPY",
    "CNY", "AUD", "CAD", "CHF", "SGD"
]

CURRENCY_NAMES = {
    "INR": "Indian Rupee",
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "JPY": "Japanese Yen",
    "CNY": "Chinese Yuan",
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CHF": "Swiss Franc",
    "SGD": "Singapore Dollar",
}

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
    "SGD": "S$",
}

CURRENCY_DETECTION_PATTERNS = {
    "INR": [r"\binr\b", r"₹", r"\brupees?\b", r"\brs\.?\b"],
    "USD": [r"\busd\b", r"\$", r"\bdollars?\b", r"\bus dollars?\b"],
    "EUR": [r"\beur\b", r"€", r"\beuros?\b"],
    "GBP": [r"\bgbp\b", r"£", r"\bpounds?\b", r"\bsterling\b"],
    "JPY": [r"\bjpy\b", r"\byen\b"],
    "CNY": [r"\bcny\b", r"\byuan\b", r"\brmb\b", r"\brenminbi\b", r"元"],
    "AUD": [r"\baud\b", r"a\$", r"\baustralian dollars?\b"],
    "CAD": [r"\bcad\b", r"c\$", r"\bcanadian dollars?\b"],
    "CHF": [r"\bchf\b", r"\bswiss francs?\b"],
    "SGD": [r"\bsgd\b", r"s\$", r"\bsingapore dollars?\b"],
}

INSIGHT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a tactical supply chain optimization assistant for {product}. "
    "Focus only on this round's order decision and explain the tradeoff between forecast coverage, "
    "carbon feasibility, and financial impact. "
    "Do not provide long-term business expansion advice, staffing advice, or generic market strategy. "
    "Output exactly 2 to 3 concise sentences."
)

CHAT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful supply chain assistant for {product}. "
    "Keep recommendations tactical and grounded in the current round context when available. "
    "Avoid unrelated long-term strategy unless the user explicitly asks for it."
)

STRICT_RETRY_SUFFIX = (
    "Strict retry: Stay on this round only. Mention at least two of forecast/order/emissions/profit/risk. "
    "No lists. No generic product portfolio advice. Keep it concise."
)

OFF_TOPIC_PATTERNS = [
    r"\bproduct categories\b",
    r"\bmost profitable products\b",
    r"\bhiring\b",
    r"\bworkforce\b",
    r"\binfrastructure\b",
    r"\blong[- ]term growth\b",
    r"\bmarketing strategy\b",
    r"\bexpand to\b",
]


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


def normalize_currency_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    value = str(code).strip().upper()
    return value if value in POPULAR_CURRENCY_CODES else None


def detect_currency_from_text(text: str) -> Optional[str]:
    normalized = str(text).lower()
    for code in POPULAR_CURRENCY_CODES:
        patterns = CURRENCY_DETECTION_PATTERNS.get(code, [])
        for pattern in patterns:
            if re.search(pattern, normalized):
                return code
    return None


def detect_currency_in_dataframe(raw_df: pd.DataFrame):
    for column_name in raw_df.columns:
        code = detect_currency_from_text(column_name)
        if code:
            return code, f"header:{column_name}"

    text_columns = list(raw_df.select_dtypes(include=["object"]).columns)
    for column_name in text_columns:
        sample_values = raw_df[column_name].dropna().astype(str).head(25)
        for value in sample_values:
            code = detect_currency_from_text(value)
            if code:
                return code, f"value:{column_name}"

    return None, None


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
        model_dtype = torch.float32 
        if device == "mps":
             model_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            SCConfig.MODEL_NAME,
            device_map=device,
            dtype=model_dtype
        )

    tokenizer = AutoTokenizer.from_pretrained(SCConfig.MODEL_NAME)
    
    log("Model Loaded Successfully.")
    return tokenizer, model


def llm_generate(
    prompt,
    tokenizer,
    model,
    max_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
):
    # Support both raw string prompts and chat messages list
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": f"You are a helpful Supply Chain Assistant optimized for {SCConfig.PRODUCT_NAME}."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt

    # Apply chat template and normalize to model.generate kwargs.
    chat_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )

    if isinstance(chat_inputs, torch.Tensor):
        model_inputs = {"input_ids": chat_inputs.to(model.device)}
    elif isinstance(chat_inputs, dict):
        model_inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in chat_inputs.items()
        }
    elif hasattr(chat_inputs, "keys"):
        model_inputs = {
            k: (chat_inputs[k].to(model.device) if hasattr(chat_inputs[k], "to") else chat_inputs[k])
            for k in chat_inputs.keys()
        }
    else:
        raise TypeError("Unsupported tokenizer chat template output type")

    prompt_input_ids = model_inputs["input_ids"]

    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            generate_kwargs.update({
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })
        outputs = model.generate(**model_inputs, **generate_kwargs)

    # Decode only the new tokens (response)
    # outputs contains [input_ids + new_tokens]
    response_ids = outputs[0][prompt_input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def get_generation_profile(mode: str = "chat", strict: bool = False) -> Dict[str, float]:
    if mode == "insight":
        if strict:
            return {
                "temperature": 0.20,
                "top_k": 20,
                "top_p": 0.70,
                "repetition_penalty": 1.15,
                "do_sample": True,
            }
        return {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.08,
            "do_sample": False,
        }

    if strict:
        return {
            "temperature": 0.30,
            "top_k": 30,
            "top_p": 0.80,
            "repetition_penalty": 1.12,
            "do_sample": True,
        }
    return {
        "temperature": 0.45,
        "top_k": 40,
        "top_p": 0.88,
        "repetition_penalty": 1.10,
        "do_sample": True,
    }


def build_round_context(
    opt_result: Dict[str, Any],
    forecast: int,
    currency_code: str,
    disruption_prob: Optional[float] = None,
    emission_factor: Optional[float] = None,
) -> Dict[str, Any]:
    financials = opt_result.get("financials", {})
    currency = normalize_currency_code(currency_code) or SCConfig.CURRENCY
    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    return {
        "product": SCConfig.PRODUCT_NAME,
        "currency_code": currency,
        "currency_symbol": symbol,
        "forecast": int(forecast),
        "optimized_qty": int(opt_result.get("optimized_qty", 0)),
        "safety_stock": int(opt_result.get("safety_stock", 0)),
        "emissions": float(opt_result.get("emissions", 0.0)),
        "carbon_cap": float(SCConfig.CARBON_CAP),
        "feasible": bool(opt_result.get("feasible", False)),
        "revenue": float(financials.get("revenue", 0.0)),
        "order_cost": float(financials.get("order_cost", 0.0)),
        "waste_cost": float(financials.get("waste_cost", 0.0)),
        "net_profit": float(financials.get("net_profit", 0.0)),
        "disruption_prob": None if disruption_prob is None else float(disruption_prob),
        "emission_factor": None if emission_factor is None else float(emission_factor),
    }


def build_insight_messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
    system_prompt = INSIGHT_SYSTEM_PROMPT_TEMPLATE.format(product=context.get("product", SCConfig.PRODUCT_NAME))

    user_content = (
        "Current round context:\n"
        f"- Forecast: {context.get('forecast')} units\n"
        f"- Recommended order: {context.get('optimized_qty')} units\n"
        f"- Emissions vs cap: {context.get('emissions'):.2f} / {context.get('carbon_cap'):.2f}\n"
        f"- Net profit: {context.get('currency_symbol')}{context.get('net_profit'):.2f} {context.get('currency_code')}\n"
        f"- Disruption risk: {context.get('disruption_prob')}\n"
        "Give a tactical recommendation for this round only."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_chat_messages(user_prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(product=SCConfig.PRODUCT_NAME)

    if context:
        system_prompt = (
            f"{system_prompt}\n"
            "Current round context:\n"
            f"forecast={context.get('forecast')} units, "
            f"order={context.get('optimized_qty')} units, "
            f"emissions={context.get('emissions'):.2f}/{context.get('carbon_cap'):.2f}, "
            f"net_profit={context.get('currency_symbol')}{context.get('net_profit'):.2f} {context.get('currency_code')}, "
            f"feasible={context.get('feasible')}, "
            f"disruption_prob={context.get('disruption_prob')}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _sentence_count(text: str) -> int:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences)


def validate_tactical_response(text: str, context: Optional[Dict[str, Any]] = None, mode: str = "insight") -> bool:
    candidate = (text or "").strip()
    if len(candidate) < 30:
        return False

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, candidate.lower()):
            return False

    decision_keywords = ["forecast", "order", "emission", "profit", "waste", "risk", "carbon", "inventory"]
    matched_keywords = sum(1 for keyword in decision_keywords if keyword in candidate.lower())

    if mode == "insight":
        if matched_keywords < 1:
            return False
        sentence_count = _sentence_count(candidate)
        if sentence_count < 1 or sentence_count > 4:
            return False
    else:
        if context and matched_keywords < 1:
            return False

    return True


def _apply_strict_retry_constraint(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages:
        return [{"role": "system", "content": STRICT_RETRY_SUFFIX}]

    strict_messages = [dict(m) for m in messages]
    if strict_messages[0].get("role") == "system":
        strict_messages[0]["content"] = f"{strict_messages[0].get('content', '')}\n{STRICT_RETRY_SUFFIX}"
        return strict_messages

    return [{"role": "system", "content": STRICT_RETRY_SUFFIX}] + strict_messages


def build_deterministic_fallback(context: Dict[str, Any]) -> str:
    symbol = context.get("currency_symbol", "")
    currency = context.get("currency_code", SCConfig.CURRENCY)
    feasible_text = "within" if context.get("feasible") else "above"
    disruption_prob = context.get("disruption_prob")
    disruption_label = "unknown"
    if disruption_prob is not None:
        disruption_label = f"{float(disruption_prob):.2f}"

    return (
        f"Recommended order is {context.get('optimized_qty')} units against a forecast of {context.get('forecast')} units for this round. "
        f"Projected net profit is {symbol}{context.get('net_profit', 0.0):.2f} {currency} and emissions are {context.get('emissions', 0.0):.2f}, which is {feasible_text} the carbon cap of {context.get('carbon_cap', 0.0):.2f}. "
        f"Risk flag: current disruption probability is {disruption_label}, so monitor actual demand and adjust safety stock next round if needed."
    )


def generate_grounded_response(
    messages: List[Dict[str, str]],
    tokenizer,
    model,
    max_tokens: int = 200,
    mode: str = "chat",
    context: Optional[Dict[str, Any]] = None,
    retry_on_failure: bool = True,
) -> Dict[str, Any]:
    profile = get_generation_profile(mode=mode, strict=False)
    effective_max_tokens = min(max_tokens, 120) if mode == "insight" else max_tokens
    first_text = llm_generate(
        messages,
        tokenizer,
        model,
        max_tokens=effective_max_tokens,
        temperature=profile["temperature"],
        top_k=int(profile["top_k"]),
        top_p=profile["top_p"],
        repetition_penalty=profile["repetition_penalty"],
        do_sample=bool(profile.get("do_sample", True)),
    )

    first_valid = validate_tactical_response(first_text, context=context, mode=mode)
    if first_valid:
        return {
            "text": first_text,
            "valid": first_valid,
            "used_retry": False,
            "used_fallback": False,
        }

    if not retry_on_failure:
        if context and mode == "insight":
            fallback_text = build_deterministic_fallback(context)
            return {
                "text": fallback_text,
                "valid": True,
                "used_retry": False,
                "used_fallback": True,
            }
        return {
            "text": first_text,
            "valid": False,
            "used_retry": False,
            "used_fallback": False,
        }

    strict_messages = _apply_strict_retry_constraint(messages)
    strict_profile = get_generation_profile(mode=mode, strict=True)
    second_text = llm_generate(
        strict_messages,
        tokenizer,
        model,
        max_tokens=effective_max_tokens,
        temperature=strict_profile["temperature"],
        top_k=int(strict_profile["top_k"]),
        top_p=strict_profile["top_p"],
        repetition_penalty=strict_profile["repetition_penalty"],
        do_sample=bool(strict_profile.get("do_sample", True)),
    )
    second_valid = validate_tactical_response(second_text, context=context, mode=mode)
    if second_valid:
        return {
            "text": second_text,
            "valid": True,
            "used_retry": True,
            "used_fallback": False,
        }

    if context and mode == "insight":
        fallback_text = build_deterministic_fallback(context)
        return {
            "text": fallback_text,
            "valid": True,
            "used_retry": True,
            "used_fallback": True,
        }

    return {
        "text": second_text,
        "valid": False,
        "used_retry": True,
        "used_fallback": False,
    }


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
    def __init__(
        self,
        num_clients: int,
        weeks: int = 52,
        data_mode: str = "synthetic",
        dataset_dir: Optional[str] = None,
        manual_currency: Optional[str] = None,
    ):
        self.num_clients = num_clients
        self.weeks = weeks
        self.data_mode = data_mode.lower()
        self.dataset_dir = dataset_dir or SCConfig.DATASET_DIR
        self.manual_currency = normalize_currency_code(manual_currency) or SCConfig.CURRENCY
        self.client_data = {}
        self.detected_currency = None
        self.currency_source = "manual"
        self.mixed_currency_detected = False
        self.currency_detection_details: Dict[str, Dict[str, str]] = {}
        self.effective_currency = self.manual_currency

        if self.data_mode == "real":
            loaded = self.load_real_data()
            if loaded < self.num_clients:
                log(f"Only {loaded} client CSV files found. Filling remaining clients with synthetic data.")
                self.generate_data(start_cid=loaded)
        else:
            self.generate_data()

        self._set_effective_currency()

    def generate_data(self, start_cid: int = 0):
        np.random.seed(42 + start_cid)
        for cid in range(start_cid, self.num_clients):
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

    def load_real_data(self) -> int:
        if not os.path.isdir(self.dataset_dir):
            log(f"Dataset directory '{self.dataset_dir}' not found. Falling back to synthetic data.")
            self._set_effective_currency()
            return 0

        csv_files = sorted([
            f for f in os.listdir(self.dataset_dir)
            if f.lower().startswith("client_") and f.lower().endswith(".csv")
        ])

        if not csv_files:
            log(f"No client CSV files found in '{self.dataset_dir}'. Falling back to synthetic data.")
            self._set_effective_currency()
            return 0

        loaded_clients = 0
        for cid, file_name in enumerate(csv_files[:self.num_clients]):
            file_path = os.path.join(self.dataset_dir, file_name)
            try:
                raw_df = pd.read_csv(file_path)
                detected_currency, detection_source = detect_currency_in_dataframe(raw_df)
                if detected_currency:
                    self.currency_detection_details[file_name] = {
                        "currency": detected_currency,
                        "source": detection_source or "unknown"
                    }
                    if self.detected_currency is None:
                        self.detected_currency = detected_currency
                    elif self.detected_currency != detected_currency:
                        self.mixed_currency_detected = True
                prepared_df = self._prepare_client_dataframe(raw_df, cid)
                self.client_data[str(cid)] = prepared_df
                loaded_clients += 1
                log(f"Loaded real client data: {file_name} ({len(prepared_df)} rows)")
            except Exception as e:
                log(f"Failed to load {file_name}: {e}")

        self._set_effective_currency()

        return loaded_clients

    def _set_effective_currency(self):
        if self.detected_currency:
            self.effective_currency = self.detected_currency
            self.currency_source = "auto-detected"
        else:
            self.effective_currency = self.manual_currency
            self.currency_source = "manual"

    def get_effective_currency(self) -> str:
        return self.effective_currency

    def get_effective_currency_symbol(self) -> str:
        return CURRENCY_SYMBOLS.get(self.effective_currency, self.effective_currency)

    def is_currency_auto_detected(self) -> bool:
        return self.currency_source == "auto-detected"

    def has_mixed_currency(self) -> bool:
        return self.mixed_currency_detected

    def _prepare_client_dataframe(self, raw_df: pd.DataFrame, cid: int) -> pd.DataFrame:
        columns = {c.lower().strip(): c for c in raw_df.columns}

        def find_col(candidates):
            for c in candidates:
                if c in columns:
                    return columns[c]
            return None

        demand_col = find_col(["demand", "quantity_sold", "sales", "units_sold"]) 
        if demand_col is None:
            raise ValueError("No demand-like column found in client dataset")

        week_col = find_col(["week"])
        date_col = find_col(["date", "timestamp"])
        disruption_col = find_col(["disruption_prob", "disruption_probability", "risk"])
        emission_col = find_col(["emission_factor", "carbon_factor", "co2e_factor"])

        out = pd.DataFrame()
        if week_col is not None:
            out["week"] = pd.to_numeric(raw_df[week_col], errors="coerce")
        else:
            out["week"] = np.arange(len(raw_df))

        if date_col is not None:
            out["date"] = pd.to_datetime(raw_df[date_col], errors="coerce")

        out["demand"] = pd.to_numeric(raw_df[demand_col], errors="coerce")

        if disruption_col is not None:
            out["disruption_prob"] = pd.to_numeric(raw_df[disruption_col], errors="coerce")
        else:
            demand_volatility = out["demand"].pct_change().abs().fillna(0.0)
            out["disruption_prob"] = np.clip(0.06 + demand_volatility * 0.8, 0.01, 0.60)

        if emission_col is not None:
            out["emission_factor"] = pd.to_numeric(raw_df[emission_col], errors="coerce")
        else:
            base = 1.20 + (0.20 * cid)
            out["emission_factor"] = np.random.normal(base, 0.08, len(out))

        out = out.dropna(subset=["demand"]).reset_index(drop=True)
        out["week"] = out["week"].fillna(pd.Series(np.arange(len(out)))).astype(int)
        out["demand"] = out["demand"].clip(lower=0).astype(int)
        out["disruption_prob"] = out["disruption_prob"].fillna(out["disruption_prob"].median()).clip(0, 1)
        out["emission_factor"] = out["emission_factor"].fillna(out["emission_factor"].median()).clip(lower=0.1)

        out = out.sort_values("week").reset_index(drop=True)
        return out

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
        self.max_vals = {}
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
        if len(data) <= self.sequence_length:
            raise ValueError(f"Client {cid} does not have enough data points for sequence length {self.sequence_length}")
        
        # Normalize Data (Simple MinMax for stability, ideally learned globally but approximating here)
        max_val = float(np.max(data)) if np.max(data) > 0 else 1.0
        self.max_vals[str(cid)] = max_val
        data_norm = data / max_val

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
                    max_val = self.max_vals.get(str(cid), float(np.max(data)) if np.max(data) > 0 else 1.0)
                    
                    # Predict last week using previous sequence
                    last_seq = data[-self.sequence_length-1:-1] / max_val
                    true_val = data[-1]
                    
                    inp = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1)
                    pred_norm = self.global_model(inp).item()
                    pred = int(pred_norm * max_val)
                    
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

    data_manager = SupplyChainDataManager(
        SCConfig.NUM_CLIENTS,
        data_mode=SCConfig.DATA_SOURCE_MODE,
        dataset_dir=SCConfig.DATASET_DIR
    )

    # Federated LSTM Training
    fed = FedSim(data_manager)
    lstm_model, metrics = fed.run(tokenizer, model)

    # FINAL FORECAST (Using trained LSTM)
    client0_df = data_manager.get_client_data("0")
    data = client0_df["demand"].values.astype(np.float32)
    max_val = fed.max_vals.get("0", float(np.max(data)) if np.max(data) > 0 else 1.0)
    
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
