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

### Requirements

This project is built upon SAM 3 and DINOv3. Please install the corresponding environments according to their official repositories:

- SAM 3: https://github.com/facebookresearch/sam3
- DINOv3: https://github.com/facebookresearch/dinov3

### Pretrained Student Weights

Baidu Cloud: XXX.

### Usage

#### 1. Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 25641 main.py --distributed --world_size 1 --epochs 100 --no-label --distillation-beta 1 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 0 1 2 3 4 5 --t-id 0 3 7 22 27 31 --model sam_vit_s --input-size 1024 --drop-path 0 --sched cosine_restart --warmup-epochs 1 --teacher-path /to/your/teacher/path/sam_vit_h_4b8939.pth --distillation-type mse --distillation-alpha 1 --data-path /to/your/dataset/path --data-set SAM --output_dir /to/your/output/path --batch-size 32 --teacher-model sam3_vit+dinov3_convnext
```

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

### Pretrained Document Localization Weights

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
