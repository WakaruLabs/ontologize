"""Training infrastructure and environment management for Ontologizer models.

This subpackage contains modules for configuring hyperparameters and metadata,
managing model state and training loops with Flax/Orbax, and serializing checkpoints:

- `config`: Dataclasses for training hyperparameters (`Hyperparams`), dataset and
  checkpoint metadata (`Metadata`), and the runtime environment (`TrainingEnv`).
- `ontostate`: Training state management (`OntoState`), optimizer initialization,
  JIT-compiled update steps, scheduled annealing (`schedules`), and the outer training loop (`train`).
- `serialize`: Orbax-based model parameter and configuration serialization and restoration.
"""
