"""Grain-based dataset loaders, sources, and preprocessing transformations.

This module provides data loading pipelines built on Google's `grain.python`:
- Transformations: `TokenizeTransform`, `FlattenTransform`, `DecodeTransform`.
- Data Sources: `NpyDataSource` (memory-mapped NumPy arrays), `JSONLDataSource`
  (O(1) random-access line-indexed JSONL), and `HFDataSource` (HuggingFace datasets).
- Loaders: `SampleLoader` (base batch loader), `EmbeddingLoader`, `ImageLoader`,
  and `TextLoader`.
"""

import grain.python as gp 
import grain.samplers as gs
import jax.numpy as jnp
import numpy as np
import einops
import json

from typing import Any, Callable, List, Dict, Iterator
from jaxtyping import Array, Float, Int

# lookup table to map HF codes to SONAR BCP-47 codes
LANG_MAP = {
   "en": "eng_Latn",
   "fr": "fra_Latn",
   "es": "spa_Latn",
   "de": "deu_Latn",
   "zh": "zho_Hans",
   # add others you want to train on...
}

class TokenizeTransform(gp.MapTransform):
    """Tokenization to be mapped over a batch of texts."""
    def __init__(self, tokenizer: Callable, maxlen: int=512):
        """Initializes the tokenizer and maximum sequence length.

        Args:
            tokenizer: HuggingFace tokenizer instance.
            maxlen: Maximum token sequence length (defaults to 512).
        """
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def map(self, item: Dict[str, str]) -> Dict[str, Int[np.ndarray, "maxlen"]]:
        """Tokenizes a single text dictionary, setting the source language dynamically.

        Looks up `item["lang"]` in `LANG_MAP` (defaulting to `"eng_Latn"`), sets `tokenizer.src_lang`,
        and tokenizes with padding to `maxlen`.

        Args:
            item: Dictionary containing at least `"text"` and `"lang"` keys.

        Returns:
            Dict[str, Int[np.ndarray, "maxlen"]]: Dictionary with `'input_ids'` and
            `'attention_mask'` NumPy arrays of shape `(maxlen,)`.
        """
        # Dynamically set the language for this specific sentence
        txt = item["text"]
        hf_lang = item["lang"]
        sonar_lang = LANG_MAP.get(hf_lang, "eng_Latn") # Default to English if unknown
        self.tokenizer.src_lang = sonar_lang

        tokens = self.tokenizer(
                txt, max_length=self.maxlen, 
                padding="max_length", truncation=True
                )
        return {
            "input_ids": np.array(tokens["input_ids"]),
            "attention_mask": np.array(tokens["attention_mask"])
        }

class FlattenTransform(gp.MapTransform):
    """Flattens a batch of images into 1-tensors. If images are in PIL format, 
    extract 'image' field and scale to (0, 1)."""
    def __init__(self, height, width):
        """Initializes the target flattened dimension from image dimensions.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
        """
        self.d = height * width

    def map(self, x):
        """Flattens image arrays or dictionary entries and normalizes pixel values.

        Extracts `'image'` or `'img'` from dictionary items, converts PIL images if needed,
        scales pixel integers to `[0.0, 1.0]`, and flattens `(..., h, w)` to `(..., h * w)`.

        Args:
            x: Image array, PIL image, or dictionary containing an image.

        Returns:
            np.ndarray: Flattened float32 NumPy array of shape `(..., height * width)`.
        """
        if isinstance(x, dict):
            # Extract the image from standard dataset dictionaries and convert PIL to numpy
            img = x.get('image', x.get('img', x))
            if hasattr(img, 'convert'):
                img = np.array(img, dtype=np.float32) / 255.0
            else:
                img = img.astype(np.float32) / 255.0
            return einops.rearrange(img, "... h w -> ... (h w)")
        return einops.rearrange(x.astype(np.float32) / 255.0, "... h w -> ... (h w)")

class DecodeTransform(gp.MapTransform):
    """Decode text batch from utf-8."""
    def map(self, x):
        """Decodes input bytes to UTF-8 if necessary.

        Args:
            x: Input string or bytes object.

        Returns:
            str: UTF-8 decoded string.
        """
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return x

class HFDataSource:
    """Grain DataSource for HuggingFace datasets. Automatically routes to 
    RandomAccessDataSource or IterDataSource depending on streaming mode."""
    def __new__(cls, dataset, text_key: str="text", lang_key: str="lang"):
        """Dispatches to a random-access or streaming Grain data source.

        Args:
            dataset: HuggingFace `Dataset` or `IterableDataset`.
            text_key: Dictionary key containing the text string (defaults to `"text"`).
            lang_key: Dictionary key containing the language identifier (defaults to `"lang"`).

        Returns:
            Union[gp.RandomAccessDataSource, _HFIterable]: Constructed Grain-compatible data source.
        """
        if hasattr(dataset, '__len__'):
            class _HFRandomAccess(gp.RandomAccessDataSource):
                def __init__(self, ds, tk, lk):
                    self.dataset = ds
                    self.text_key = tk
                    self.lang_key = lk
                def __len__(self): return len(self.dataset)
                def __getitem__(self, idx):
                    return {"text": self.dataset[int(idx)][self.text_key],
                            "lang": self.dataset[int(idx)].get(self.lang_key, "en")}
            return _HFRandomAccess(dataset, text_key, lang_key)
        else:
            class _HFIterable:
                def __init__(self, ds, tk, lk):
                    self.dataset = ds
                    self.text_key = tk
                    self.lang_key = lk
                def __iter__(self):
                    iterator = iter(self.dataset)
                    while True:
                        try:
                            item = next(iterator)
                            yield {"text": item[self.text_key],
                                   "lang": item.get(self.lang_key, "en")}
                        except StopIteration:
                            break
                        except Exception as e:
                            # Skip corrupted lines (e.g. pyarrow.lib.ArrowInvalid)
                            print(f"Skipping corrupted sample: {e}")
                            continue
            return _HFIterable(dataset, text_key, lang_key)

class NpyDataSource(gp.RandomAccessDataSource):
    """Grain DataSource with O(1) random access into a `.npy` array on disk
    (e.g. an embedding cache written by `encode_corpus.py`). The array is
    memory-mapped lazily on first access so the source pickles cleanly into
    grain worker processes."""
    def __init__(self, file_path):
        """Initializes the memory-mapped NumPy array data source.

        Args:
            file_path: Path string or Path object to the `.npy` file.
        """
        self.file_path = str(file_path)
        self._arr = np.load(self.file_path, mmap_mode="r")
        self._shape = self._arr.shape

    def __len__(self):
        """Returns the total number of samples along the leading axis.

        Returns:
            int: Sample count (`self._shape[0]`).
        """
        return self._shape[0]

    def __getitem__(self, idx):
        """Retrieves a single sample array by index.

        Copies the row out of the memory-mapped buffer to prevent pinning memmaps in worker processes.

        Args:
            idx: Integer sample index.

        Returns:
            np.ndarray: Sample array at index `idx`.
        """
        if self._arr is None:
            self._arr = np.load(self.file_path, mmap_mode="r")
        # copy the row out so batches don't pin the memmap
        return np.array(self._arr[int(idx)])

    def __getstate__(self):
        """Custom pickle serialization state excluding the active mmap array reference.

        Returns:
            dict: State dictionary with `_arr` reset to None for clean process pickling.
        """
        return {"file_path": self.file_path, "_shape": self._shape,
                "_arr": None}


class JSONLDataSource(gp.RandomAccessDataSource):
   """Grain DataSource for streaming text from JSONL files with random access.

   Scans the target file at initialization to build an in-memory index of byte
   offsets for each newline, enabling O(1) random-access seeking during batch sampling.
   """
   def __init__(self, file_path: str, text_key: str="text"):
       """Indexes line offsets for O(1) random access into the JSONL file.

       Args:
           file_path: Path string to the `.jsonl` file.
           text_key: JSON key containing the text payload (defaults to `"text"`).
       """
       self.file_path = file_path
       self.text_key = text_key
       # Index the line offsets for O(1) random access to any sample
       self._offsets = []
       with open(file_path, 'rb') as f:
           offset = 0
           for line in f:
               self._offsets.append(offset)
               offset += len(line)

   def __len__(self):
       """Returns the total number of lines indexed in the file.

       Returns:
           int: Total sample count.
       """
       return len(self._offsets)

   def __getitem__(self, index):
       """Seeks to the indexed byte offset, reads and parses the JSON line.

       Args:
           index: Integer sample index.

       Returns:
           str: Extracted text string for the sample.
       """
       # Jump directly to the line in the file
       with open(self.file_path, 'rb') as f:
           f.seek(self._offsets[index])
           line = f.readline()
           data = json.loads(line)
           return data[self.text_key]


class SampleLoader:
    """Generic data loader class that does not apply additional transformations
    to a batch. Used when `Metadata.srctype="embedding"`."""
    def __init__(self, b: int, epochs: int, d: int, src, operations, 
                 threads: int=0, shuffle: bool=False, seed: int=42,
                 drop_remainder: bool=True):
        """Initializes the Grain data loader pipeline.

        Args:
            b: Batch size integer.
            epochs: Number of complete epochs over the dataset.
            d: Target feature or token sequence dimension.
            src: Data source object (`RandomAccessDataSource` or iterable).
            operations: List of Grain transformation operations to apply before batching.
            threads: Worker process count for parallel loading (default 0).
            shuffle: If True, shuffles sample indices across epochs (default False).
            seed: Random seed for indexing and shuffling (default 42).
            drop_remainder: If True, drops batches smaller than `b` at epoch boundaries (default True).
        """
        self.d = d
        self.b = b
        self.threads = threads
        self.seed = seed
        self.src = src 
        self.transformations = operations + [
                gp.Batch(batch_size=b, drop_remainder=drop_remainder)
        ]
        
        if hasattr(self.src, '__len__'):
            self.sampler = gs.IndexSampler(
                    num_records=len(self.src),
                    shard_options=gp.ShardOptions(shard_index=0, shard_count=1),
                    shuffle=shuffle,
                    seed=seed,
                    num_epochs=epochs
                    )
            self.loader = gp.DataLoader(
                    data_source=self.src,
                    operations=self.transformations,
                    sampler=self.sampler,
                    worker_count=self.threads,
                    shard_options=gp.ShardOptions(shard_index=0, shard_count=1)
                    )
        else:
            self.sampler = None
            self.loader = None

    def __iter__(self) -> Iterator[Dict[str, Int[np.ndarray, "b d"]]]:
        """Yields batches of samples from the underlying Grain loader or streaming generator.

        Returns:
            Iterator[Dict[str, Int[np.ndarray, "b d"]]]: Iterator yielding batched NumPy arrays or dictionaries.
        """
        if self.loader is not None:
            return iter(self.loader)
        
        # Manual pipeline evaluation for Iterable datasets
        iterator = iter(self.src)
        for op in self.transformations:
            if hasattr(op, 'map'):
                def _make_mapper(mapper, it):
                    return (mapper.map(x) for x in it)
                iterator = _make_mapper(op, iterator)
            elif hasattr(op, 'filter'):
                def _make_filter(f, it):
                    return (x for x in it if f.filter(x))
                iterator = _make_filter(op, iterator)
            elif isinstance(op, gp.Batch):
                def _batch(it, b, drop):
                    batch = []
                    for item in it:
                        batch.append(item)
                        if len(batch) == b:
                            # Collate dictionaries of arrays/scalars
                            if isinstance(batch[0], dict):
                                yield {k: np.stack([b_i[k] for b_i in batch]) for k in batch[0]}
                            else:
                                yield np.stack(batch)
                            batch = []
                    if not drop and batch:
                        if isinstance(batch[0], dict):
                            yield {k: np.stack([b_i[k] for b_i in batch]) for k in batch[0]}
                        else:
                            yield np.stack(batch)
                iterator = _batch(iterator, op.batch_size, op.drop_remainder)
        return iterator

class EmbeddingLoader(SampleLoader):
    """Loader for precomputed embedding vectors (e.g. a cached corpus from
    `encode_corpus.py`); applies no per-sample transformations. Used when
    `Metadata.srctype="embedding"`."""
    def __init__(self, b: int, epochs: int, src, d: int=0,
                 threads: int=0, shuffle: bool=False, seed: int=42,
                 drop_remainder: bool=True):
        """Initializes the embedding loader.

        Args:
            b: Batch size integer.
            epochs: Number of training epochs.
            src: Embedding data source (e.g., `NpyDataSource`).
            d: Embedding vector dimension (defaults to 0).
            threads: Background worker process count (defaults to 0).
            shuffle: Whether to shuffle embeddings across epochs (defaults to False).
            seed: Random seed for shuffling (defaults to 42).
            drop_remainder: Whether to drop partial batches (defaults to True).
        """
        super().__init__(b, epochs, d, src, [],
                         threads, shuffle, seed, drop_remainder)

class ImageLoader(SampleLoader):
    """Loader that applies `FlattenTransform` to a batch. Used when
    `Metadata.srctype="image"`."""
    def __init__(self, b: int, epochs: int, src, height: int, width: int,
                 threads: int=0, shuffle: bool=False, seed: int=42,
                 drop_remainder: bool=True):
        """Initializes the image loader with flattening and normalization transforms.

        Args:
            b: Batch size integer.
            epochs: Number of training epochs.
            src: Image data source.
            height: Image pixel height.
            width: Image pixel width.
            threads: Background worker process count (defaults to 0).
            shuffle: Whether to shuffle images across epochs (defaults to False).
            seed: Random seed for shuffling (defaults to 42).
            drop_remainder: Whether to drop partial batches (defaults to True).
        """
        d = height * width
        operations = [
                FlattenTransform(height, width)
                ]
        super().__init__(b, epochs, d, src, operations, 
                         threads, shuffle, seed, drop_remainder)

class TextLoader(SampleLoader):
    """Loader that applies `DecodeTransform` and `TokenizeTransform` to a batch.
    Used when `Metadata.srctype="text"`"""
    def __init__(self, b: int, epochs: int, src, tokenizer: Callable,
                 threads: int=0, shuffle: bool=False, maxlen: int=512, seed: int=42,
                 drop_remainder: bool=True):
        """Initializes the text loader with decoding and tokenization transformations.

        Args:
            b: Batch size integer.
            epochs: Number of training epochs.
            src: Text data source.
            tokenizer: HuggingFace tokenizer callable or instance.
            threads: Background worker process count (defaults to 0).
            shuffle: Whether to shuffle text samples across epochs (defaults to False).
            maxlen: Maximum token sequence length (defaults to 512).
            seed: Random seed for shuffling (defaults to 42).
            drop_remainder: Whether to drop partial batches (defaults to True).
        """
        operations = [
                DecodeTransform(),
                TokenizeTransform(tokenizer, maxlen=maxlen),
                ]
        super().__init__(b, epochs, maxlen, src, operations, 
                         threads, shuffle, seed, drop_remainder)
