## Image Commenting Project

### Introduction
This repo is the PyTorch implementation of Image Commenting Project that has three modules, image-to-comment generator, comment reranker and inference pipeline. Our generator module utilizes an adaptive attention based seq2seq architecture to generate a plausible response for a given image, which consists of a ResNet-based visual encoder and a LSTM-based decoder. The key innovation is to implement an adaptive attention mechanism (originally introduced in [Lu et al., CVPR'17](https://arxiv.org/pdf/1612.01887.pdf)) into the decoding process that can flexibly decide when to look at which region of the image or just follow the Language Model to predict the next word at each time-step. Separately introducing a image-comment matching model is to alleviate the "safe response" issue by reranking the generated comment candidates, while simultaneously ensuring the promising relevance.

### Requirements
- Python 3.6
- PyTorch 0.4
- Java 1.8
- [COCO-API](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)
