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
from ontologize.training.serialize import restore_spec
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
    model = Ontologizer(**restore_spec(manager, step))

    state_dict = manager.restore(step, items={'state': None})['state']
    
    if 'opt_state' in state_dict:
        trained_params = state_dict['params']
    else:
        trained_params = state_dict
        
    while 'params' in trained_params:
        trained_params = trained_params['params']
        
    print(f"Successfully loaded step {step}.")

    # 4. Reference norm: the decoder expects embeddings at the scale of the raw
    # mean-pooled encoder output, not unit norm (see decode.py). Dictionary
    # entries have no original norm of their own, so use the corpus's typical
    # one (textfid.SONAR_NORM).
    from textfid import SONAR_NORM
    ref_norm = SONAR_NORM
    print(f"Reference embedding norm: {ref_norm:.4f}")

    # 5. Decode Tags
    print("\n--- Decoding Tags ---")

    # Get embeddings for entries and uniform tags
    R_entries, _ = model.apply({'params': trained_params}, method=Ontologizer.decodeEntries)
    R_uniform = model.apply({'params': trained_params}, method=Ontologizer.decodeUniform)

    def decode_embeddings(jax_embs, batch_size=16):
        import torch.nn.functional as F
        pt_embs = t.from_dlpack(jax_embs).to(device, dtype=t.float32)
        pt_embs = F.normalize(pt_embs, p=2, dim=-1) * ref_norm
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        
        all_texts = []
        for i in range(0, len(pt_embs), batch_size):
            batch = pt_embs[i:i+batch_size]
            
            mock_encoder_outputs = BaseModelOutput(last_hidden_state=batch.unsqueeze(1))
            
            generated_ids = pt_decoder.generate(
                encoder_outputs=mock_encoder_outputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=64,
                num_beams=4,
                repetition_penalty=1.2
            )
            
            texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(texts)
            print(f"Decoded {len(all_texts)} / {len(pt_embs)}...", end="\r")
        print()
        return all_texts

    print("Decoding Entries...")
    entries_texts = decode_embeddings(R_entries)
    
    print("Decoding Uniform...")
    uniform_texts = decode_embeddings(R_uniform)
    
    entries_path = checkpoint_dir / "entries.txt"
    with open(entries_path, "w", encoding="utf-8") as f:
        for txt in entries_texts:
            f.write(txt.replace('\n', ' ') + "\n")
            
    uniform_path = checkpoint_dir / "uniform.txt"
    with open(uniform_path, "w", encoding="utf-8") as f:
        for txt in uniform_texts:
            f.write(txt.replace('\n', ' ') + "\n")
            
    print(f"Saved decoded texts to {entries_path} and {uniform_path}")

if __name__ == "__main__":
    main()
