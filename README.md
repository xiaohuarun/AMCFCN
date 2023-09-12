# AMCFCN
Significant progress is being made in the field of multi-view clustering.For multi-view clustering, the key is to obtain a view-common representation of a set of view data.However, the existing literature faces certain limitations.

#Environment

- Python 3.9.7
- PyTorch 1.8.0
- CUDA 11.4

# Training

All our experiments are put in `./experiments`, data files under `data/processed`.

Note: Before you run the program firstly, you should run `datatool/load_dataset` to generate dataset.

You could quickly run our experiments by: `python train.py -c [config name]`.

For example: `python train.py -mnist`
