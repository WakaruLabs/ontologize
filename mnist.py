from datasets import load_dataset
from pathlib import Path

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import OntoState
from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
from ontologize.training.data import ImageLoader

def main():
    height = 28
    width = 28

    d_in = height * width
    d_out = 10

    e_enc = 256
    e_dec = 256
    k = 10
    h = 5
    n_l = 1

    b = 2000
    epochs = 100

    lr = 0.001
    wd = 0.0
    stddev = 0.4

    s_g = 0.001
    s_L1K = 0.001
    s_L1F = 0.001

    path = Path("data").resolve()
    out = path / "out/mnist/L1"
    save_each = 10
    checkpoint_each = 10000
    threads = 4

    model = Ontologizer(d_in, d_in, e_enc, e_dec, k, h, n_l)
    hyper = Hyperparams(d_in, d_in, b, lr, wd, stddev,
                        s_g=s_g, s_L1K=s_L1K, s_L1F=s_L1F)
    meta = Metadata("image", out_path=out, epochs=epochs, threads=threads,
                    save_each=save_each, checkpoint_each=checkpoint_each)
    env = TrainingEnv(model, hyper, meta, 
                      kwargs_loader={"height": height, "width": width})

    src = load_dataset("mnist", split="train")
    state = env.train(src)
    print("Training Complete!")

if __name__ == "__main__":
    main()
