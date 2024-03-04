# AMCFCN


# Abstract

Advances in deep learning have propelled the evolution of multi-view clustering techniques, which strive to obtain a view-common representation from multi-view datasets. However, the contemporary multi-view clustering community confronts two prominent challenges. One is that view-specific representations lack guarantees to reduce noise introduction, and another is that the fusion process compromises view-specific representations, resulting in the inability to capture efficient information from multi-view data. This may negatively affect the accuracy of the clustering results. In this paper, we introduce a novel technique named the "contrastive attentive strategy" to address the above problems. Our approach effectively extracts robust view-specific representations from multi-view data with reduced noise while preserving view completeness. This results in the extraction of consistent representations from multi-view data while preserving the features of view-specific representations. We integrate view-specific encoders, a hybrid attentive module, a fusion module, and deep clustering into a unified framework called AMCFCN. Experimental results on four multi-view datasets demonstrate that our method, AMCFCN, outperforms seven competitive multi-view clustering methods. 

# Architecture

![model](https://github.com/xiaohuarun/AMCFCN/img/model.jpg](https://github.com/xiaohuarun/AMCFCN/blob/main/img/model.jpg)

# Environment

- Python 3.9.7
- PyTorch 1.8.0
- CUDA 11.4

# Training

All our experiments are put in `./experiments`, data files under `data/processed`.

Note: Before you run the program firstly, you should run `datatool/load_dataset` to generate dataset.

You could quickly run our experiments by: `python train.py -c [config name]`.

For example: `python train.py -mnist`
