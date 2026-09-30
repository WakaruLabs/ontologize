# disable preallocation so jax and torch can share VRAM
import os
import sys

# Inject LD_LIBRARY_PATH via process replacement to bypass shell sanitation
if "CUDNN_INJECTED" not in os.environ:
    try:
        import nvidia.cudnn
        cudnn_lib = list(nvidia.cudnn.__path__)[0] + "/lib"
        old_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{old_path}"
        os.environ["CUDNN_INJECTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except ImportError:
        pass

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from pathlib import Path

# MUST import JAX before PyTorch to ensure JAX loads its newer CuDNN first
import jax
import jax.numpy as jnp
# Force JAX backend initialization immediately
jax.random.PRNGKey(0)

import torch as t
import optax
import orbax.checkpoint as ocp

from transformers import AutoTokenizer, AutoModel, M2M100ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from ontologize.ontologizer import Ontologizer
from ontologize.training.serialize import migrate_spec
from ontologize.training.ontostate import OntoState, load_params
from ontologize.data.pretrained import encode

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Decode Ontologizer embeddings")
    parser.add_argument("checkpoint", type=str, nargs="?", default="data/out/sonar/linear",
                        help="Path to the checkpoint directory (e.g. data/out/sonar/linear)")
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # 1. Load the Encoders and Decoders (PyTorch)
    print("Loading PyTorch models...")
    encoder_id = "cointegrated/SONAR_200_text_encoder"
    decoder_id = "raxtemur/SONAR_200_text_decoder"

    from ontologize.data.pretrained import pretrained_transformer
    pt_encoder, tokenizer = pretrained_transformer(encoder_id, dtype_str="float32", dev=device)
    
    # The fix to prevent dataset collapse!
    tokenizer.src_lang = "eng_Latn"
    
    pt_decoder = M2M100ForConditionalGeneration.from_pretrained(decoder_id).to(device)

    # 3. Load the Checkpoint
    checkpoint_dir = Path(args.checkpoint).resolve()
    print(f"Loading checkpoint from {checkpoint_dir}...")
    manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers={
            'state': ocp.PyTreeCheckpointer(),
            'spec': ocp.PyTreeCheckpointer()
        }
    )
    # Restore the state
    step = manager.latest_step()
    if step is None:
        raise ValueError("No checkpoint found in directory!")

    print("Setting up JAX Ontologizer from checkpoint spec...")
    spec = manager.restore(step, items={'spec': None})['spec']
    model = Ontologizer(**migrate_spec(spec))

    state_dict = manager.restore(step, items={'state': None})['state']
    
    if 'opt_state' in state_dict:
        trained_params = state_dict['params']
    else:
        trained_params = state_dict
        
    if 'params' not in trained_params:
        trained_params = {'params': trained_params}
    while 'params' in trained_params and 'params' in trained_params['params']:
        trained_params = trained_params['params']
        
    print(f"Successfully loaded step {step}.")

    from ontologize.chat import ChatEnv
    chat_env = ChatEnv(model)

    # 4. Interactive Decoding Loop
    while True:
        text, arglist = chat_env.interact()
        if text is None:
            break
            
        # A. Encode text to PyTorch tensor
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        
        # B. Get SONAR Embedding
        jax_embedding = encode(pt_encoder, inputs, device)
        
        # Calculate original norm
        with t.no_grad():
            outputs = pt_encoder(**{k: v.to(device) for k, v in inputs.items()})
            E_pt = outputs.last_hidden_state
            mask_pt = inputs['attention_mask'].to(device)
            mask = mask_pt.unsqueeze(-1).float()
            E_pooled = t.sum(E_pt * mask, dim=1)
            n_token = t.sum(mask, dim=1).clamp(min=1e-9)
            E_mean = E_pooled / n_token
            orig_norm = t.norm(E_mean, p=2, dim=-1, keepdim=True)
        
        # C. Pass through Ontologizer withArgs
        R_reconstructed, _, _, _ = model.apply(
            trained_params, jax_embedding, arglist=arglist, 
            temperature=0.1, method=Ontologizer.withArgs
        )

        
        # Calculate and print MSE
        mse = float(jnp.mean((jax_embedding - R_reconstructed)**2))
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # D. Convert and decode
        R_pt = t.from_dlpack(R_reconstructed).to(device, dtype=t.float32)
        import torch.nn.functional as F
        R_pt = F.normalize(R_pt, p=2, dim=-1) * orig_norm
        
        R_pt_seq = R_pt.unsqueeze(1)
        mock_encoder_outputs = BaseModelOutput(last_hidden_state=R_pt_seq)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        
        generated_ids = pt_decoder.generate(
            encoder_outputs=mock_encoder_outputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=4,
            repetition_penalty=1.2
        )
        
        reconstructed_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Reconstructed: {reconstructed_text}")

if __name__ == "__main__":
    main()
