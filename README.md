# pytorch-pwc

This is a personal fork of [this implementation](https://github.com/sniklaus/pytorch-pwc) of PWC-net for optical flow estimation. For more information check out the original repository or [read the paper](https://arxiv.org/abs/1709.02371).

In this fork I've refactored the structure so that the network is separated and can be imported on it's own. I've also added a `video_inference.py` function so that video's can be fed into the optical flow model.

![](images/crowd_flow.png)

## Installation

After installing pytorch and the necessary dependencies with e.g. conda, the rest can be installed with 

```bash
pip install -r requirements.txt
```

Once this is done the pretrained network weights can be downloaded with

```bash
./download.sh
```