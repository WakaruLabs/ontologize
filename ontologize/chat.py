"""Interactive command-line REPL for dynamic ontofeature interventions.

This module provides the `ChatEnv` class, an interactive terminal command loop
allowing users to configure, inspect, and reset causal interventions on trained
`Ontologizer` models. Interventions (setting, adding, subtracting, zeroing, or
scaling dictionary features) are staged per layer as `DictIntervention` dataclass
instances and passed to `Ontologizer.withArgs` for steered text generation or analysis.
"""
import jax.numpy as jnp
from typing import List, Optional, Tuple
from ontologize.ontologizer import Ontologizer, DictIntervention
from dataclasses import replace

class ChatEnv:
    """An interactive CLI environment for setting `Ontologizer.withArgs` interventions."""
    
    def __init__(self, model: Ontologizer):
        """Initialize the chat environment with empty interventions for all layers.

        Args:
            model: `Ontologizer` model instance to configure interventions for.
        """
        self.model = model
        self.num_layers = model.l
        self.interventions = [DictIntervention() for _ in range(self.num_layers)]

    def _print_state(self):
        """Print currently active non-empty interventions across all layers to stdout."""
        print("\n=== Current Interventions ===")
        has_active = False
        for i, inter in enumerate(self.interventions):
            active = []
            for field in inter.__dataclass_fields__:
                val = getattr(inter, field)
                if val is not None:
                    active.append(f"{field}={val.tolist() if hasattr(val, 'tolist') else val}")
            if active:
                print(f"Layer {i}: " + ", ".join(active))
                has_active = True
        if not has_active:
            print("No active interventions.")
        print("=============================\n")

    def _parse_array(self, val_str: str, dtype):
        """Parse a comma-separated string of numeric values into a JAX array.

        Args:
            val_str: Comma-separated string of integers or floats (e.g. "5,10" or "0.5,1.0").
            dtype: Target numeric type (`float` or `int`).

        Returns:
            JAX array of dtype `float32` or `int32`, or `None` if string is empty.
        """
        if not val_str.strip():
            return None
        # Handle comma-separated list of numbers
        values = [float(x) if dtype == float else int(x) for x in val_str.split(',')]
        return jnp.array(values, dtype=jnp.float32 if dtype == float else jnp.int32)

    def interact(self) -> Tuple[Optional[str], Optional[List[DictIntervention]]]:
        """
        Interactive loop for setting interventions.
        Returns a tuple of (text_to_process, list_of_interventions).
        """
        print("Entering ChatEnv... Type 'help' for commands.")
        while True:
            self._print_state()
            try:
                cmd_line = input("ChatEnv> ").strip()
                if not cmd_line:
                    continue
                
                cmd = cmd_line.split()
                action = cmd[0].lower()
                
                if action in ['quit', 'q', 'exit']:
                    return None, None
                    
                elif action == 'help':
                    print("Commands:")
                    print("  set <layer> <field> <val1,val2,...>  # e.g., set 0 k_add 5,10")
                    print("  clear <layer>                        # clear interventions for a layer")
                    print("  clear_all                            # clear all interventions")
                    print("  run <text>                           # run the model with current interventions")
                    print("  q, quit, exit                        # exit")
                    
                elif action == 'clear_all':
                    self.interventions = [DictIntervention() for _ in range(self.num_layers)]
                    print("Cleared all interventions.")
                    
                elif action == 'clear':
                    if len(cmd) < 2:
                        print("Usage: clear <layer>")
                        continue
                    layer = int(cmd[1])
                    if 0 <= layer < self.num_layers:
                        self.interventions[layer] = DictIntervention()
                        print(f"Cleared interventions for layer {layer}.")
                    else:
                        print(f"Invalid layer: {layer}. Must be 0 to {self.num_layers-1}.")
                        
                elif action == 'set':
                    if len(cmd) < 4:
                        print("Usage: set <layer> <field> <val1,val2,...>")
                        continue
                    layer = int(cmd[1])
                    field = cmd[2]
                    val_str = "".join(cmd[3:])
                    
                    if not (0 <= layer < self.num_layers):
                        print(f"Invalid layer: {layer}. Must be 0 to {self.num_layers-1}.")
                        continue
                        
                    if not hasattr(DictIntervention, field):
                        print(f"Unknown field: {field}")
                        print("Available fields:", ", ".join(DictIntervention.__dataclass_fields__.keys()))
                        continue
                        
                    dtype = float if field == 'scale' else int
                    try:
                        val_arr = self._parse_array(val_str, dtype)
                        kwargs = {field: val_arr}
                        self.interventions[layer] = replace(self.interventions[layer], **kwargs)
                        print(f"Set {field} on layer {layer} to {val_arr}")
                    except ValueError as e:
                        print(f"Failed to parse values: {e}. Ensure they are comma-separated numbers.")
                        
                elif action == 'run':
                    if len(cmd) < 2:
                        print("Usage: run <text to process>")
                        continue
                    text = " ".join(cmd[1:])
                    # Return a copy of the interventions
                    return text, list(self.interventions)
                else:
                    print(f"Unknown command: {action}")
            except (EOFError, KeyboardInterrupt):
                return None, None
            except Exception as e:
                print(f"Error: {e}")
