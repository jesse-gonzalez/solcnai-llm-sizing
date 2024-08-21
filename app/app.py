import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import (
    DATA_TYPES,
    PARAMETERS,
    OPTIMIZERS,
    load_predefined_models,
)
from utils import (
    calculate_inference_memory,
    calculate_training_memory,
)


# ----------------- Streamlit Setup ----------------- #
st.set_page_config(page_title="LLM Memory Requirements")
st.title("LLM Memory Requirements")


# ----------------- Sidebar Initialization ----------------- #
MODELS = load_predefined_models()

def set_values():
    """Update the values based on the selected model"""
    if st.session_state.model in MODELS:
        model_info = MODELS[st.session_state.model]
        for param in PARAMETERS:
            if PARAMETERS[param] in model_info:
                st.session_state[param] = model_info[PARAMETERS[param]]
            else:
                st.session_state[param] = None
    else:
        for param in PARAMETERS:
            st.session_state[param] = None


def get_gpu_specs():
    return pd.read_csv("data/gpu_specs.csv")

def show_gpu_info(info, vendor=""):
    for var in ['Inference']:
        _info = info.loc[var]
        if _info['Number of GPUs'] >= 3:
            func = st.error
            icon = "⛔"
        elif _info['Number of GPUs'] == 2:
            func = st.warning
            icon = "⚠️"
        else:
            func = st.success
            icon = "✅"
        
        msg = f"You require **{_info['Number of GPUs']}** GPUs for **{var}**"
        func(msg, icon=icon)

def get_name(index):
    row = gpu_specs.iloc[index]
    return f"{row['Product Name']} ({row['RAM (GB)']} GB, {row['Year']})"

# ----------------- Sidebar UI ----------------- #
# Model Selection
model = st.sidebar.selectbox(
    "Model", list(MODELS.keys()), index=None, on_change=set_values, key="model"
)

# Parameters
model_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=None,
    key="model_size",
    help="Number of parameters in the model in billions",
)
precision = st.sidebar.selectbox(
    "Precision",
    DATA_TYPES,
    index=None,
    key="precision",
    help="Data type used (int 8 and int 4 are for quantization)",
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=1,
    key="batch_size",
)
sequence_length = st.sidebar.number_input(
    "Sequence Length",
    min_value=0,
    step=1,
    value=2048,
    key="sequence_length",
    help="Number of tokens in the input sequence.",
)
hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=None,
    key="hidden_size",
    help="Size of the hidden layer (given by the model card).",
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=None,
    key="num_hidden_layers",
    help="Number of layers in the model (given by the model card).",
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
)

gpu_specs = get_gpu_specs()
gpu_vendor = st.sidebar.selectbox("GPU Vendor", ["NVIDIA"])
gpu_info = gpu_specs[gpu_specs['Vendor'] == gpu_vendor].sort_values('Product Name')[::-1]

# Set the RAM filter to show a range from 0 to 100, defaulting to 24 to 80
min_ram = gpu_info['RAM (GB)'].min()
max_ram = gpu_info['RAM (GB)'].max()
ram = st.sidebar.slider("Filter by RAM (GB)", 0.0, 100.0, (min_ram, max_ram), step=0.5)

gpu_info = gpu_info[gpu_info["RAM (GB)"].between(ram[0], ram[1])]
gpu = st.sidebar.selectbox("GPU", gpu_info['Product Name'].index.tolist(), format_func=lambda x : gpu_specs.iloc[x]['Product Name'])
gpu_spec = gpu_specs.iloc[gpu]
gpu_spec.name = 'INFO'

st.sidebar.dataframe(gpu_spec.T.astype(str))

# ----------------- Main Screen UI ----------------- #
# Dividing the screen into two tabs
#inference, training = st.tabs(["Inference", "Training"])
hardware, inference  = st.tabs(["GPU Info", "Inference"])

# Tab 2: Training
#training1, training2 = training.columns(2)
#optimizer = training2.selectbox("Optimizer", list(OPTIMIZERS.keys()), key="optimizer")
#trainable_parameters = training2.slider(
#    "Percentage of trainable parameters", 0, 100, 100, key="trainable_params"
#)

# Inference Memory
inference_memory = calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
)

with inference:
    inference.write(f"**Total Inference Memory**: {inference_memory['inference_memory']}")
    inference.write(f"- **Model Weights**: {inference_memory['model_weights']}")
    inference.write(f"- **KV Cache**: {inference_memory['kv_cache']}")
    inference.write(f"- **Activation Memory**: {inference_memory['activation_memory']}")

# Calculate the number of GPUs required
#total_inference_memory = inference_memory['inference_memory']
#gpu_memory_capacity = gpu_specs.iloc[gpu]['RAM (GB)']
#print(gpu_memory_capacity)  # Assuming 'Memory Capacity' is the column name
#num_gpus_required = (total_inference_memory / gpu_memory_capacity) + 1  # Adding 1 to ensure we round up

with hardware:
    hardware.write(f"**GPU**: {gpu_specs.iloc[gpu]['Product Name']}")
    #hardware.write(f"**Number of GPUs Required**: {num_gpus_required}")

# Training Memory
# training_memory = calculate_training_memory(
#     model_size,
#     precision,
#     batch_size,
#     sequence_length,
#     hidden_size,
#     num_hidden_layers,
#     num_attention_heads,
#     optimizer,
#     trainable_parameters,
# )

# training1.write(f"**Total Training Memory**: {training_memory['training_memory']}")
# training1.write(f"- **Model Weights**: {training_memory['model_weights']}")
# training1.write(f"- **KV Cache**: {training_memory['kv_cache']}")
# training1.write(f"- **Activation Memory**: {training_memory['activation_memory']}")
# training1.write(f"- **Optimizer Memory**: {training_memory['optimizer_memory']}")
# training1.write(f"- **Gradients Memory**: {training_memory['gradients_memory']}")

# ----------------- Error Handling ----------------- #
if None in st.session_state.values():
    st.warning("Some information is missing.")
