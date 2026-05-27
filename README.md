<p align="center">

  <h1 align="center">RDLNet++: Frequency-Decoupled Multi-Teacher Distillation for Real-World Document Localization</h1>

**RDLNet++** is an enhanced network for real-world document localization. The project aims to accurately detect document regions in challenging mobile-captured images, including complex backgrounds, diverse document categories, low contrast, shadows, occlusion, and perspective distortion. RDLNet++ introduces a Frequency-Decoupled Multi-Teacher Distillation strategy to transfer knowledge from large scale vision foundation models to a lightweight encoder. It further refines feature representations through a transformer decoder and performs precise document localization using dedicated classification, mask, and corner point prediction branches, together with a Boundary Constraint Strategy. This repository also provides the **RWMD-Extended** dataset, an expanded version of the Real-World Mobile Document dataset. RWMD-Extended contains more diverse document categories and challenging real-world scenarios, supporting more rigorous evaluation of document localization methods. Experiments on multiple datasets show that RDLNet++ achieves stable and competitive localization performance while maintaining an efficient model design. 

## News

## Table of Contents
- [Introduction](#introduction)
- [Contributions](#contributions)
- [RWMD-Extended Dataset](#rwmd-extended-dataset)
- [Knowledge Distillation](#knowledge-distillation)
- [Document Localization](#document-localization)

## Introduction
RDLNet++ is a lightweight real-world document localization network that uses Frequency-Decoupled Multi-Teacher Distillation, a transformer decoder, dedicated output branches, and a Boundary Constraint Strategy to achieve accurate and efficient document localization.

RWMD-Extended is an expanded real-world mobile document dataset containing 2,526 images across eleven document categories, with challenging cases such as low contrast, complex illumination, cluttered backgrounds, and ambiguous document boundaries.

## Contributions
  - We propose a novel Frequency-Decoupled Multi-Teacher Distillation (FD-MTD), along with detailed explanations and visualization analyses.
  - We propose a triple-branch decoder, which is improved by introducing the Boundary Constraint Strategy (BCS).
  - We propose a RWMD-Extended dataset by incorporating more challenging scenarios to the RWMD dataset.
  - Extensive experiments on multiple public benchmarks demonstrate that our method achieves higher Jaccard Index scores than most state-of-the-art methods.
    
## RWMD-Extended Dataset
RWMD-Extended is an expanded version of the Real-World Mobile Document (RWMD) dataset. It contains 2,526 mobile-captured images from real-world environments, including 517 newly added images and eleven document categories.
The dataset includes challenging cases such as low contrast, complex illumination, cluttered backgrounds, and ambiguous document boundaries. Each image provides instance-level segmentation masks, category labels, and corner point coordinates for document localization research.

Baidu Cloud: XXX.

## Knowledge Distillation

## Document Localization

### Requirements

This project follows the environment setup of the original Mask2Former based implementation. The environment was tested with the following configuration:

- Python 3.8
- PyTorch 1.9.0 with CUDA 11.1
- torchvision 0.10.0 with CUDA 11.1
- Detectron2 compatible with PyTorch 1.9 and CUDA 11.1

Please install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

### Pretrained Weights

Baidu Cloud: XXX.

### Usage

#### 1. Training

```bash
python train_net.py
```

#### 2. Testing with pretrained weights

```bash
python demo/demo3.py
```
