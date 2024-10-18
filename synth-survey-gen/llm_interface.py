import torch
import json
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig, LlamaConfig


model_dir = Path("meta/Llama3.2-1B/")

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_dir)  # Adjust path as necessary

# Load the config from params.json and manually set attributes
with open(model_dir / "params.json") as f:
    config_params = json.load(f)

# Create the LlamaConfig with custom parameters
config = LlamaConfig(
    hidden_size=config_params['dim'],                      # corresponds to 'dim' in your JSON
    intermediate_size=int(config_params['dim'] * config_params['ffn_dim_multiplier']),  # ffn_dim_multiplier
    num_attention_heads=config_params['n_heads'],          # corresponds to 'n_heads'
    num_hidden_layers=config_params['n_layers'],           # corresponds to 'n_layers'
    vocab_size=config_params['vocab_size'],                # corresponds to 'vocab_size'
    rms_norm_eps=config_params['norm_eps'],                # corresponds to 'norm_eps'
    max_position_embeddings=2048,                          # Default value (you can change this)
    num_key_value_heads=config_params['n_kv_heads'],       # corresponds to 'n_kv_heads'
    rope_theta=config_params['rope_theta'],                # corresponds to 'rope_theta'
    use_scaled_rope=config_params['use_scaled_rope'],      # corresponds to 'use_scaled_rope'
)


# Create the model with the loaded configuration
model = LlamaForCausalLM(config)

# Load the state_dict from consolidated.00.pth
state_dict = torch.load(model_dir / "consolidated.00.pth", map_location="cpu")

# Load the weights into the model
model.load_state_dict(state_dict)

# Run a sample input through the model
input_text = "Hello, I'm testing the LLaMA model!"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=50)

# Decode the generated output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
