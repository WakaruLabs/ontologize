import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List, Dict, Any, Tuple
from ontologize.jax.layers.dictblock import DictBlock

# NOTE: The DictInterpreter is inherently stateful/PyTorch based in its interactions 
# with HuggingFace models. A full JAX translation would require using JAX/Flax 
# equivalents of the HuggingFace models, or separating the interpretation step
# out of the purely functional `nn.Module`. For now, we maintain the interface.

class InterpretableDictBlock(DictBlock):
    """
    A DictBlock with added interpretation capabilities.
    """
    interpreter_model: Optional[str] = None
    
    def setup(self):
        super().setup()
        # In a strict Flax module, storing mutable state like an interpreter model
        # object directly on the module is an anti-pattern. Interpretation should
        # ideally happen outside the forward pass or be injected explicitly.
        # This is left as a placeholder for architectural equivalence.
        self.interpreter = None

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(X)
    
    def forward_with_interpretation(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Normal forward pass plus interpretation of what happened.
        """
        output = self(X)
        interpretation = {}
        if self.interpreter is not None:
            # We would need a JAX compatible interpreter here
            interpretation = self.interpreter.analyze_activation_path(X, output)
        return output, interpretation
