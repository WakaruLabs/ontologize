import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from ontologize.jax.layers.dictenc import DictEnc
from ontologize.jax.layers.dictblock import DictBlock
from ontologize.jax.layers.dict_interpreter import DictInterpreter

class ModelExemplarAnalyzer:
    """
    Analyzer for generating and managing exemplars for trained models.
    Unlike the PyTorch version, this does not mutate the model instance.
    Instead, it orchestrates the model, its parameters, and the interpreter.
    """
    
    def __init__(
        self, 
        model: DictEnc, 
        model_params: dict,
        interpreter_model: str = "gpt2"
    ):
        """
        Args:
            model: DictEnc model to analyze (blueprint)
            model_params: The learned parameters for the model
            interpreter_model: Language model for generating exemplars
        """
        self.model = model
        self.model_params = model_params
        
        # Extract dictionary params assuming standard Flax parameter naming
        try:
            self.dict_params = model_params['params']['dict']['dicts']
        except KeyError:
            raise ValueError("Could not find 'dict' parameters in the provided model_params.")
            
        self.interpreter = DictInterpreter(
            dict_size=model.dictSize,
            embedding_dim=model.embedding2,
            heads=model.heads,
            dict_params=self.dict_params,
            decoder_model=interpreter_model
        )
    
    def extract_exemplars(self, max_length: int = 50) -> List[str]:
        """
        Generate exemplar texts for all dictionary entries.
        
        Returns:
            List of exemplar descriptions
        """
        return self.interpreter.generate_all_exemplars(max_length)
    
    def analyze_sample(
        self,
        sample_input: jnp.ndarray,
        return_exemplars: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze how a sample input activates dictionary entries.
        
        Args:
            sample_input: Input tensor to analyze
            return_exemplars: Whether to include exemplar texts
            
        Returns:
            Analysis dictionary with activated entries and their roles
        """
        # Run through encoder (using model.apply)
        # We need to extract just the encoder function
        def encode_fn(params, x):
            return self.model.bind({'params': params['params']}).encoder(x)
            
        def dict_cluster_fn(params, x):
            return self.model.bind({'params': params['params']}).dict.cluster(x)
            
        def dict_forward_fn(params, x):
            return self.model.bind({'params': params['params']}).dict(x)
            
        def decode_fn(params, x):
            return self.model.bind({'params': params['params']}).decoder(x)
        
        encoded = encode_fn(self.model_params, sample_input)
        dict_output = dict_forward_fn(self.model_params, encoded)
        
        # Get clustering weights
        K = dict_cluster_fn(self.model_params, encoded)  # [batch, k, h]
        K_avg = K.mean(axis=(0, 2))  # Average across batch and heads
        
        # Find most active entries
        top_k = min(10, self.model.dictSize)
        top_indices = jnp.argsort(K_avg)[-top_k:][::-1]
        top_weights = K_avg[top_indices]
        
        active_entries = []
        for idx, weight in zip(top_indices, top_weights):
            if weight > 0.01:  # Threshold for "active"  
                entry_info = {
                    "index": int(idx),
                    "activation": float(weight)
                }
                
                if return_exemplars:
                    entry_info["exemplar"] = self.interpreter.generate_exemplar(int(idx))
                
                active_entries.append(entry_info)
        
        # Calculate reconstruction
        reconstructed = decode_fn(self.model_params, dict_output)
        reconstruction_error = float(jnp.linalg.norm(reconstructed - sample_input))
        
        return {
            "active_entries": active_entries,
            "num_active": len(active_entries),
            "sparsity": len(active_entries) / self.model.dictSize,
            "reconstruction_error": reconstruction_error
        }
    
    def compare_samples(
        self,
        samples: List[jnp.ndarray]
    ) -> Dict[str, Any]:
        """
        Compare which dictionary entries are activated by different samples.
        """
        all_analyses = []
        all_active_indices = []
        
        for sample in samples:
            analysis = self.analyze_sample(sample, return_exemplars=False)
            all_analyses.append(analysis)
            all_active_indices.append(
                set(entry["index"] for entry in analysis["active_entries"])
            )
        
        # Find shared and unique entries
        if len(all_active_indices) > 1:
            shared = set.intersection(*all_active_indices)
            unique_per_sample = [
                indices - shared for indices in all_active_indices
            ]
        else:
            shared = all_active_indices[0] if all_active_indices else set()
            unique_per_sample = [set()]
        
        # Get exemplars for shared entries
        shared_exemplars = {}
        for idx in shared:
            shared_exemplars[idx] = self.interpreter.generate_exemplar(idx)
        
        return {
            "analyses": all_analyses,
            "shared_entries": list(shared),
            "shared_exemplars": shared_exemplars,
            "unique_entries_per_sample": [list(u) for u in unique_per_sample]
        }
    
    def save_analysis_report(
        self,
        output_path: Path,
        sample_inputs: Optional[List[jnp.ndarray]] = None
    ):
        """
        Save a comprehensive analysis report.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating exemplars for all dictionary entries...")
        exemplars = self.extract_exemplars()
        
        report = {
            "model_info": {
                "dict_size": self.model.dictSize,
                "embedding_dim": self.model.embedding2,
                "heads": self.model.heads
            },
            "exemplars": exemplars
        }
        
        if sample_inputs:
            print(f"Analyzing {len(sample_inputs)} sample inputs...")
            sample_analyses = []
            
            for i, sample in enumerate(sample_inputs):
                analysis = self.analyze_sample(sample)
                sample_analyses.append({
                    "sample_id": i,
                    **analysis
                })
            
            report["sample_analyses"] = sample_analyses
            
            if len(sample_inputs) > 1:
                comparison = self.compare_samples(sample_inputs)
                report["comparison"] = comparison
        
        with open(output_path / "exemplar_analysis.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        with open(output_path / "exemplar_analysis.txt", 'w') as f:
            f.write("EXEMPLAR ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL INFORMATION:\n")
            f.write(f"  Dictionary size: {report['model_info']['dict_size']}\n")
            f.write(f"  Embedding dimension: {report['model_info']['embedding_dim']}\n")
            f.write(f"  Number of heads: {report['model_info']['heads']}\n\n")
            
            f.write("DICTIONARY EXEMPLARS:\n")
            f.write("-" * 40 + "\n")
            for i, exemplar in enumerate(exemplars[:30]): 
                f.write(f"{i:3d}: {exemplar}\n")
            
            if sample_inputs and "sample_analyses" in report:
                f.write("\n\nSAMPLE ANALYSES:\n")
                f.write("-" * 40 + "\n")
                for analysis in report["sample_analyses"]:
                    f.write(f"\nSample {analysis['sample_id']}:\n")
                    f.write(f"  Active entries: {analysis['num_active']}\n")
                    f.write(f"  Sparsity: {analysis['sparsity']:.3f}\n")
                    f.write(f"  Reconstruction error: {analysis['reconstruction_error']:.4f}\n")
                    f.write("  Top activated entries:\n")
                    for entry in analysis["active_entries"][:5]:
                        f.write(f"    #{entry['index']}: {entry['exemplar'][:50]}... ")
                        f.write(f"(activation: {entry['activation']:.3f})\n")
        
        print(f"✓ Analysis report saved to {output_path}")
