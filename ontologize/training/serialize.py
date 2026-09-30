import orbax.checkpoint as ocp
import flax.linen as nn
from typing import Any, Tuple, Dict
import dataclasses

#: spec keys earlier checkpoints wrote under other names. A model is saved
#: as `dataclasses.asdict(model)` and restored by splatting that back into
#: its class, so renaming a field makes every checkpoint written before the
#: rename unconstructable -- `105223c` renamed these three and broke every
#: pre-existing checkpoint for every script that loads one. Mapping them
#: forward here keeps that a one-line cost rather than a per-caller one.
LEGACY_SPEC_KEYS = {
    "activation_scale": "activation_router",
    "gate_scale": "gate_router",
    "biased_scale": "biased_router",
}


def migrate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A saved model spec with legacy field names mapped forward.

    Renames only. A key this does not know is left alone, so a genuinely
    unrecognized one still raises from the constructor instead of being
    dropped: silently discarding a spec key would build a model whose
    configuration differs from the one that was trained, which is worse
    than failing to build one at all."""
    spec = dict(spec)
    for old, new in LEGACY_SPEC_KEYS.items():
        if old in spec:
            spec[new] = spec.pop(old)
    return spec

def save_model(
    checkpoint_dir: str, 
    step: int, 
    params_pytree: Any, 
    model: nn.Module
):
    """
    Saves Flax model weights and configuration using Orbax.
    
    Args:
        checkpoint_dir: Directory to save the checkpoints.
        step: Training step or epoch.
        params_pytree: The Flax variables/params PyTree.
        model: The Flax nn.Module instance, used to extract configuration metadata.
    """
    # Configure Orbax CheckpointManager
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpointer = ocp.CheckpointManager(
        checkpoint_dir, 
        options=options
    )
    
    # Extract module configuration as dictionary
    # Note: this assumes the module only has simple, JSON-serializable dataclass fields
    config_metadata = dataclasses.asdict(model)
    
    # Save both weights and metadata
    save_args = ocp.args.StandardSave(params_pytree)
    checkpointer.save(
        step, 
        args=save_args,
        metadata=config_metadata 
    )
    print(f"Saved checkpoint to {checkpoint_dir} at step {step}")

def load_model(
    checkpoint_dir: str, 
    step: int, 
    model_class: type[nn.Module],
    dummy_input: Any = None
) -> Tuple[Any, nn.Module]:
    """
    Loads Flax model weights and reinstantiates the model from Orbax.
    
    Args:
        checkpoint_dir: Directory containing the checkpoints.
        step: Step to load.
        model_class: The class of the Flax nn.Module to instantiate.
        dummy_input: Optional dummy input to initialize the skeleton if needed.
                     Not strictly required if orbax handles unstructured restore, 
                     but recommended for safe structured restore.
                     
    Returns:
        Tuple of (restored_params, instantiated_model)
    """
    checkpointer = ocp.CheckpointManager(checkpoint_dir)
    
    # 1. Load the metadata to reconstruct the model blueprint
    metadata = checkpointer.metadata(step)
    if metadata is None:
        raise ValueError(f"No metadata found in checkpoint at step {step}")
        
    model = model_class(**migrate_spec(metadata))
    
    # 2. Restore the parameters
    # If dummy_input is provided, we can do a structured restore for safety
    if dummy_input is not None:
        import jax
        key = jax.random.PRNGKey(0)
        skeleton = model.init(key, dummy_input)
        restore_args = ocp.args.StandardRestore(skeleton)
        restored = checkpointer.restore(step, args=restore_args)
    else:
        # Unstructured restore (just loads the PyTree as saved)
        restore_args = ocp.args.StandardRestore()
        restored = checkpointer.restore(step, args=restore_args)
        
    print(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
    return restored, model
