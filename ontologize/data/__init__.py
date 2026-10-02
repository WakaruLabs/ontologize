"""Data loading, dataset pipelines, and pretrained encoder integrations for Ontologizer.

This subpackage contains modules for loading, transforming, and streaming datasets:

- `langs`: Language code lookup mappings between HuggingFace dataset identifiers (ISO 639)
  and SONAR/NLLB BCP-47 language codes (`MC4_TO_SONAR`).
- `loaders`: Grain data sources (`NpyDataSource`, `JSONLDataSource`, `HFDataSource`),
  transformation maps (`TokenizeTransform`, `FlattenTransform`, `DecodeTransform`), and
  batch loaders (`SampleLoader`, `EmbeddingLoader`, `ImageLoader`, `TextLoader`).
- `multilingual`: Helpers for downloading, filtering, and interleaving multilingual
  datasets from HuggingFace (`load_lang`, `load_langs`, `mc4_data`).
- `pretrained`: PyTorch transformer integration, tokenization, masked L2 sentence-level
  mean pooling (`l2_pooling`), DLPack zero-copy tensor sharing (`encode`), and autoregressive
  sequence decoding (`decode`).
"""
