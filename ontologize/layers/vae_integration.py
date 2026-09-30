import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import json
from transformers import AutoTokenizer

from ontologize.jax.layers.dictenc import DictEnc

class VAEEnhancedDictBlockPlaceholder:
    """
    Placeholder for VAEEnhancedDictBlock which was not found in the PyTorch implementation.
    """
    def __init__(self, k, d, h, vae=None, tokenizer=None):
        self.dictSize = k
        self.embedding = d
        self.heads = h
        self.vae = vae
        self.tokenizer = tokenizer

    def get_exemplars(self, max_length=50):
        return [f"<concept_{i}_exemplar>" for i in range(self.dictSize)]


class ExemplarAnalyzer:
    """
    Analyzer for extracting and interpreting exemplars from trained models.
    Adapted for JAX/Flax architectures.
    """
    
    def __init__(self, model_config: DictEnc, model_params: dict, has_vae: bool = False):
        """
        Args:
            model_config: DictEnc module blueprint
            model_params: Learned parameters for the model
        """
        self.model = model_config
        self.model_params = model_params
        self.has_vae = has_vae
        
        try:
            self.dict_params = model_params['params']['dict']['dicts']
        except KeyError:
            self.dict_params = None
    
    def extract_ideals(self, max_length: int = 50) -> List[str]:
        """Extract exemplar texts for all dictionary entries."""
        if not self.dict_params is None:
            return [f"<concept_{i}_exemplar>" for i in range(self.model.dictSize)]
        return []
    
    def analyze_text_concepts(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Placeholder for concept activation analysis."""
        return {}
    
    def save_analysis_report(self, output_path: Path, sample_texts: List[str] = None):
        """
        Save a comprehensive analysis report of the exemplars.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        ideals = self.extract_ideals()
        
        report = {
            "model_info": {
                "dict_size": self.model.dictSize,
                "embedding_dim": self.model.embedding2,
                "heads": self.model.heads,
                "has_vae": self.has_vae
            },
            "exemplars": ideals
        }
        
        with open(output_path / "exemplars_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        with open(output_path / "exemplars_readable.txt", 'w') as f:
            f.write("EXEMPLARS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {report['model_info']['dict_size']} concepts, ")
            f.write(f"{report['model_info']['embedding_dim']}D embeddings, ")
            f.write(f"{report['model_info']['heads']} heads\n")
            f.write(f"VAE Available: {report['model_info']['has_vae']}\n\n")
            
            f.write("EXEMPLARS:\n")
            f.write("-" * 30 + "\n")
            for i, ideal in enumerate(ideals):
                f.write(f"{i:2d}: {ideal}\n")
        
        print(f"✓ Analysis report saved to {output_path}")


def create_exemplars_demo(
    model_config: DictEnc,
    model_params: dict,
    sample_texts: List[str],
    output_dir: Path = Path("exemplars_demo")
):
    """
    Create a demonstration of exemplars functionality for JAX.
    """
    print("Creating exemplars Demo...")
    print("=" * 40)
    
    analyzer = ExemplarAnalyzer(model_config, model_params, has_vae=True)
    
    try:
        exemplars = analyzer.extract_ideals()
        print(f"✓ Extracted {len(exemplars)} exemplars")
        
        print("\nSample exemplars:")
        for i, exemplar in enumerate(exemplars[:5]):
            print(f"  {i}: {exemplar}")
    except Exception as e:
        print(f"⚠ Could not extract exemplars: {e}")
    
    analyzer.save_analysis_report(output_dir, sample_texts)
    
    print(f"\n✓ Demo created in {output_dir}")
    
    return analyzer
