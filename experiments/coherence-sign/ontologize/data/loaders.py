import grain.python as gp 
import grain.samplers as gs
import jax.numpy as jnp
import numpy as np
import einops
import json

from typing import Any, Callable, List, Dict, Iterator
from jaxtyping import Array, Float, Int

# lookup table to map HF codes to the FLORES-200 codes SONAR's tokenizer
# takes. Only these five are mapped: `TokenizeTransform` tags every other
# language `eng_Latn`.
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
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def map(self, item: Dict[str, str]) -> Dict[str, Int[np.ndarray, "maxlen"]]:
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
        self.d = height * width

    def map(self, x):
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
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return x

class HFDataSource:
    """Grain DataSource for HuggingFace datasets. Automatically routes to 
    RandomAccessDataSource or IterDataSource depending on streaming mode."""
    def __new__(cls, dataset, text_key: str="text", lang_key: str="lang"):
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
    memory-mapped on construction, dropped when the source is pickled into a
    grain worker process, and mapped again there on first access."""
    def __init__(self, file_path):
        self.file_path = str(file_path)
        self._arr = np.load(self.file_path, mmap_mode="r")
        self._shape = self._arr.shape

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        if self._arr is None:
            self._arr = np.load(self.file_path, mmap_mode="r")
        # copy the row out so batches don't pin the memmap
        return np.array(self._arr[int(idx)])

    def __getstate__(self):
        return {"file_path": self.file_path, "_shape": self._shape,
                "_arr": None}


class JSONLDataSource(gp.RandomAccessDataSource):
   """Grain DataSource for streaming text from JSONL files with random access."""
   def __init__(self, file_path: str, text_key: str="text"):
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
       return len(self._offsets)

   def __getitem__(self, index):
       # Jump directly to the line in the file
       with open(self.file_path, 'rb') as f:
           f.seek(self._offsets[index])
           line = f.readline()
           data = json.loads(line)
           return data[self.text_key]


class SampleLoader:
    """Generic data loader class: applies `operations`, then batches. A sized
    source is read through a grain `IndexSampler` (shuffled with `seed` when
    `shuffle`, repeated for `epochs`); an iterable one is evaluated manually.
    The subclasses below choose the operations; `Metadata.loader` falls back
    to this class for an unknown `srctype`."""
    def __init__(self, b: int, epochs: int, d: int, src, operations, 
                 threads: int=0, shuffle: bool=False, seed: int=42,
                 drop_remainder: bool=True):
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
        super().__init__(b, epochs, d, src, [],
                         threads, shuffle, seed, drop_remainder)

class ImageLoader(SampleLoader):
    """Loader that applies `FlattenTransform` to a batch. Used when
    `Metadata.srctype="image"`."""
    def __init__(self, b: int, epochs: int, src, height: int, width: int,
                 threads: int=0, shuffle: bool=False, seed: int=42,
                 drop_remainder: bool=True):
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
        operations = [
                DecodeTransform(),
                TokenizeTransform(tokenizer, maxlen=maxlen),
                ]
        super().__init__(b, epochs, maxlen, src, operations, 
                         threads, shuffle, seed, drop_remainder)
