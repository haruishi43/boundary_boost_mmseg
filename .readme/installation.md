# Installation

NOTE: Please use Python3.8 for now (Have not tested on other versions, might work on 3.9 and above).

## Setups

### MMSegmentation

This project is built-on `mmseg==0.27.0`.
`mmseg==1.x.x` is not supported yet.
Please follow the [official installation guide](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) to install `mmseg`.
If you are not familiar with `mmseg` or other OpenMM-Lab related frameworks, it may be easier to follow some of the guidelines first:
- [Setting up the dataset](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md):
  - we will use Cityscapes for training and validation
- [Training](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/train.md)
  - make sure that the training script works


### PyEdgeEval

This project uses `pyEdgeEval` for data preprocessing pipelines (OTF GT generation for boundaries).
It's currently only tested on Python 3.8 for `pip` installable package, but it can also be installed from source (currently closed-source, but hopefully opened soon).

```Bash
# for python3.8
pip install pyEdgeEval

# to install from source
# WIP
```

### Install other dependencies

```Bash
pip install -r requirements.txt
```
