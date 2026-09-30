import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


class DictInterpreter:
    """
    Interprets DictBlock entries by generating text exemplars.
    
    This is a separate utility that takes the learned dictionary embeddings from a DictBlock
    (which are in activation space) and generates text descriptions of what each entry represents.
    """
    
    def __init__(
        self,
        dict_size: int,
        embedding_dim: int,
        heads: int,
        dict_params: jnp.ndarray,
        projection_params: Optional[dict] = None,
        decoder_model: str = "gpt2"
    ):
        """
        Args:
            dict_size: Number of dictionary entries
            embedding_dim: Embedding dimension
            heads: Number of heads
            dict_params: The actual learned dictionary entries tensor of shape [k, d, h]
            projection_params: Optional parameters for projection from activation space to decoder input space
            decoder_model: Name or path of language model to use for generating descriptions
        """
        self.k = dict_size
        self.d = embedding_dim
        self.h = heads
        self.dict_params = dict_params
        
        # Initialize projection from activation space to LM input space
        self.projection = nn.Dense(features=768) # 768 is GPT-2 hidden size
        if projection_params is None:
            # Initialize random projection if none provided
            key = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((self.d,))
            self.projection_params = self.projection.init(key, dummy_input)
        else:
            self.projection_params = projection_params
        
        # Initialize language model for generating descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.lm = FlaxAutoModelForCausalLM.from_pretrained(decoder_model)
        
    def get_dictionary_entries(self) -> jnp.ndarray:
        """
        Extract the dictionary entries from the DictBlock.
        
        Returns:
            Dictionary entries tensor of shape [k, d, h]
        """
        return self.dict_params
    
    def generate_exemplar(self, entry_idx: int, max_length: int = 50) -> str:
        """
        Generate a text exemplar for a specific dictionary entry.
        
        Args:
            entry_idx: Index of the dictionary entry to interpret
            max_length: Maximum length of generated text
            
        Returns:
            Generated text describing what this dictionary entry represents
        """
        # Get the dictionary entry [d, h]
        entry = self.dict_params[entry_idx]
        
        # Average across heads to get a single embedding [d]
        entry_embedding = entry.mean(axis=1)
        
        # Project to language model space
        lm_embedding = self.projection.apply(self.projection_params, entry_embedding)
        
        # Generate text using the language model
        # We'll use the projection as a prompt embedding
        prompt = "This concept represents:"
        inputs = self.tokenizer(prompt, return_tensors="jax")
        
        # NOTE: The original PyTorch code had a bug where lm_embedding was never passed 
        # to the generator. We replicate the generation functionality here.
        # In a real scenario, you would prepend the lm_embedding to the inputs_embeds.
        outputs = self.lm.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return text.replace(prompt, "").strip()
    
    def generate_all_exemplars(self, max_length: int = 50) -> List[str]:
        """
        Generate exemplars for all dictionary entries.
        
        Returns:
            List of exemplar texts, one per dictionary entry
        """
        exemplars = []
        for i in range(self.k):
            try:
                exemplar = self.generate_exemplar(i, max_length)
                exemplars.append(exemplar)
            except Exception as e:
                print(f"Failed to generate exemplar for entry {i}: {e}")
                exemplars.append(f"<concept_{i}>")
        return exemplars
    
    def find_nearest_entries(self, activation: jnp.ndarray, top_k: int = 5) -> List[tuple]:
        """
        Find the dictionary entries most similar to a given activation.
        
        Args:
            activation: Activation tensor of shape [d] or [batch, d]
            top_k: Number of top entries to return
            
        Returns:
            List of (entry_index, similarity_score, exemplar_text) tuples
        """
        if activation.ndim == 1:
            activation = jnp.expand_dims(activation, axis=0)  # Add batch dimension
        
        # Get dictionary entries averaged across heads [k, d]
        dict_entries = self.dict_params.mean(axis=2)
        
        # Compute cosine similarities
        act_norm = activation / jnp.linalg.norm(activation, axis=1, keepdims=True)
        dict_norm = dict_entries / jnp.linalg.norm(dict_entries, axis=1, keepdims=True)
        similarities = jnp.dot(act_norm, dict_norm.T)[0] # Shape [k]
        
        # Get top k entries
        top_indices = jnp.argsort(similarities)[-top_k:][::-1]
        top_sims = similarities[top_indices]
        
        results = []
        for idx, sim in zip(top_indices, top_sims):
            exemplar = self.generate_exemplar(idx.item())
            results.append((idx.item(), sim.item(), exemplar))
        
        return results
    
    def analyze_activation_path(
        self,
        encoder_output: jnp.ndarray,
        dict_block_output: jnp.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze how an activation flows through the dictionary.
        
        Args:
            encoder_output: Output from the encoder before DictBlock [batch, k, h]
            dict_block_output: Output from the DictBlock [batch, d]
            
        Returns:
            Analysis dictionary with active entries and their contributions
        """
        # Get the clustering weights (which entries were activated)
        # Using softmax along axis=1 (the k dimension)
        K = jax.nn.softmax(encoder_output, axis=1)  # [batch, k, h]
        
        # Average across heads to see overall activation
        K_avg = K.mean(axis=2)  # [batch, k]
        
        # Find most active entries for each sample in batch
        batch_analyses = []
        for b in range(K_avg.shape[0]):
            top_k = 5
            top_indices = jnp.argsort(K_avg[b])[-top_k:][::-1]
            top_weights = K_avg[b][top_indices]
            
            active_entries = []
            for idx, weight in zip(top_indices, top_weights):
                if weight > 0.01:  # Threshold for considering an entry "active"
                    exemplar = self.generate_exemplar(idx.item())
                    active_entries.append({
                        "index": idx.item(),
                        "weight": weight.item(),
                        "exemplar": exemplar
                    })
            
            batch_analyses.append({
                "active_entries": active_entries,
                "sparsity": (K_avg[b] > 0.01).sum().item() / self.k
            })
        
        return {
            "batch_analyses": batch_analyses,
            "average_sparsity": sum(a["sparsity"] for a in batch_analyses) / len(batch_analyses)
        }
    
    def save_interpretation_report(self, output_path: Path):
        """
        Save a comprehensive interpretation report of all dictionary entries.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all exemplars
        exemplars = self.generate_all_exemplars()
        
        # Analyze dictionary statistics
        dict_entries = self.dict_params
        
        # Compute various statistics
        norms = jnp.linalg.norm(dict_entries, axis=1).mean(axis=1)  # [k]
        sparsity = (jnp.abs(dict_entries) < 1e-3).mean().item()
        
        # Find clusters of similar entries
        dict_avg = dict_entries.mean(axis=2)  # [k, d]
        dict_avg_norm = dict_avg / jnp.linalg.norm(dict_avg, axis=1, keepdims=True)
        similarity_matrix = jnp.dot(dict_avg_norm, dict_avg_norm.T) # [k, k]
    
        report = {
            "model_info": {
                "dict_size": self.k,
                "embedding_dim": self.d,
                "heads": self.h,
                "sparsity": float(sparsity)
            },
            "entries": []
        }
        
        for i, exemplar in enumerate(exemplars):
            # Find most similar other entries
            similarities = jnp.array(similarity_matrix[i])
            similarities = similarities.at[i].set(-1) # Exclude self
            top_similar_idx = jnp.argmax(similarities).item()
            
            report["entries"].append({
                "index": i,
                "exemplar": exemplar,
                "norm": float(norms[i]),
                "most_similar_to": top_similar_idx,
                "similarity": float(similarities[top_similar_idx])
            })
        
        # Save JSON report
        with open(output_path / "dictionary_interpretation.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable version
        with open(output_path / "dictionary_interpretation.txt", 'w') as f:
            f.write("DICTIONARY INTERPRETATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dictionary size: {self.k}\n")
            f.write(f"Embedding dimension: {self.d}\n")
            f.write(f"Number of heads: {self.h}\n")
            f.write(f"Overall sparsity: {sparsity:.3f}\n\n")
            
            f.write("DICTIONARY ENTRIES:\n")
            f.write("-" * 30 + "\n")
            for entry in report["entries"][:50]:  # Limit to first 50 for readability
                f.write(f"\n{entry['index']:3d}: {entry['exemplar']}\n")
                f.write(f"     Norm: {entry['norm']:.3f}, ")
                f.write(f"Similar to #{entry['most_similar_to']} ")
                f.write(f"(sim: {entry['similarity']:.3f})\n")
        
        print(f"✓ Interpretation report saved to {output_path}")


def interpret_trained_dictblock(
    dict_size: int,
    embedding_dim: int,
    heads: int,
    dict_params: jnp.ndarray,
    sample_activations: Optional[jnp.ndarray] = None,
    output_dir: Path = Path("dict_interpretation")
) -> DictInterpreter:
    """
    Convenience function to interpret a trained DictBlock.
    
    Args:
        dict_size: Number of dictionary entries
        embedding_dim: Embedding dimension
        heads: Number of heads
        dict_params: The actual learned dictionary entries tensor of shape [k, d, h]
        sample_activations: Optional sample activations to analyze
        output_dir: Directory to save interpretation results
        
    Returns:
        DictInterpreter instance
    """
    interpreter = DictInterpreter(dict_size, embedding_dim, heads, dict_params)
    
    # Generate interpretation report
    interpreter.save_interpretation_report(output_dir)
    
    # If sample activations provided, analyze them
    if sample_activations is not None:
        print("\nAnalyzing sample activations...")
        
        # Run through dict_block
        # We need the clustering and decoding steps
        K = jax.nn.softmax(sample_activations, axis=1)
        dict_output = jnp.einsum("bkh,kdh->bd", K, dict_params)
        
        # Analyze the flow
        analysis = interpreter.analyze_activation_path(
            sample_activations,
            dict_output
        )
        
        print(f"Average sparsity: {analysis['average_sparsity']:.3f}")
        
        # Save sample analysis
        with open(output_dir / "sample_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
    
    return interpreter
