# Functions for integrating pretrained PyTorch transformers.

import torch as t
import jax.numpy as jnp
from typing import Any
from jaxtyping import Float, Array, Int

from transformers import AutoTokenizer, AutoModel

def get_torch_dtype(name: str) -> Any:
    """Fetches `torch` dtype from a `dtype_str`."""
    dtypes = {
        "float32": t.float32,
        "float16": t.float16,
        "bfloat16": t.bfloat16,
        "float64": t.float64,
    }
    return dtypes.get(name.lower(), t.float32)

def pretrained_transformer(model_id: str, dtype_str: str="bfloat16", dev=t.device("cuda")):
    """Load pretrained HuggingFace model."""
    dtype = get_torch_dtype(dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load in bfloat16 to save 50% VRAM (1.2GB instead of 2.4GB)
    
    if "SONAR" in model_id:
        from transformers import M2M100Config
        from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
        from huggingface_hub import hf_hub_download
        
        config = M2M100Config.from_pretrained(model_id)
        path = hf_hub_download(model_id, "pytorch_model.bin")
        state_dict = t.load(path, map_location="cpu", weights_only=True)
        encoder = M2M100Encoder(config)
        encoder.load_state_dict(state_dict)
        encoder = encoder.to(dtype)
    else:
        model = AutoModel.from_pretrained(model_id, dtype=dtype)
        encoder = model.encoder

    if dev is not None:
        encoder = encoder.to(dev)

    return encoder, tokenizer

def tokenize(tokenizer, batch, *args, **kwargs):
    return tokenizer(batch, return_tensors="pt", padding=True, *args, **kwargs)

def l2_pooling(
   E: Float[Array, "batch seq_len d_model"],
   attention_mask: Int[Array, "batch seq_len"]
) -> Float[Array, "batch d_model"]:
   """
   Performs masked mean pooling and L2 normalization to generate 
   sentence-level embeddings.
   """
   # 1. Expand mask for element-wise multiplication [batch, seq, 1]
   mask = jnp.expand_dims(attention_mask, axis=-1).astype(E.dtype)
   
   # 2. Mask out padding tokens and sum across the sequence
   E_pooled = jnp.sum(E * mask, axis=1)
   
   # 3. Count non-padding tokens (clamped to avoid division by zero)
   n_token = jnp.sum(mask, axis=1)
   n_token = jnp.clip(n_token, min=1e-9)
   
   # 4. Mean Pooling
   E_mean = E_pooled / n_token 
   
   # 5. L2 Normalization (SONAR vectors are unit-normed)
   # Using a small epsilon for numerical stability
   norm = jnp.linalg.norm(E_mean, axis=-1, keepdims=True)
   E_seq = E_mean / jnp.clip(norm, min=1e-9)
   
   return E_seq

def encode(model, inputs, dev=t.device("cuda")):
    """Encode a tokenized batch from a data loader. Rund `model(**inputs)` and
    pools a model's `last_hidden_state` with `l2_pooling`."""
    # Grain provides numpy arrays, but the encoder needs PyTorch tensors
    inputs = {k: t.as_tensor(v) for k, v in inputs.items()}
    if dev is not None:
        inputs = {k: v.to(dev) for k, v in inputs.items()}
    with t.no_grad():
        outputs = model(**inputs)
        E_pt = outputs.last_hidden_state
        mask_pt = inputs['attention_mask']
        E = jnp.from_dlpack(E_pt)
        mask = jnp.from_dlpack(mask_pt)
        return l2_pooling(E, mask)

def decode(model, outputs, E: Float[Array, "... b d"], *args, **kwargs):
    """Decode an `Ontologizer` output. Requires the model to have a `generate` method."""
    with t.no_grad():
        E_pt = t.from_dlpack(E).unsqueeze(1)
        outputs["last_hidden_state"] = E_pt
        return model.generate(
                encoder_outputs=outputs, *args, **kwargs
                ).sequences

