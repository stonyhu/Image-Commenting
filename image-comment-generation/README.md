## Setup
- `cd data && bash download.sh`
- install python 3.6 and pytorch 0.4
- install java 1.8
- install [COCOAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)
- `python prepro.py`

## train
```
mkdir -p output
python train.py --tensorboard
```