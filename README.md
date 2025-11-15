# DOSTrack - Dynamic Object Tracking with Adaptive Sizing

**DOSTrack** addresses fundamental limitations in visual object tracking by introducing **truly dynamic template and search region sizing**. Unlike traditional trackers that use fixed 128×128 templates and 256×256 search regions, DOSTrack dynamically adapts to object characteristics, solving critical problems where:
- Small objects (30×30 pixels) get poor representation in large templates
- Large objects exceed fixed search regions
- Objects moving out-of-view require adaptive search expansion

Built on DINOv3's powerful feature extraction and RoPE positional embeddings, DOSTrack achieves robust tracking across extreme scale variations and challenging scenarios.

## 🎯 Core Innovations

### **1. Adaptive Template & Search Sizing**
```
Object Size         Template Size    Search Size
───────────────────────────────────────────────
Small (< 50px)      64×64           192×192
Medium (50-150px)   128×128         256×256
Large (150-300px)   224×224         448×448
Very Large (>300px) 384×384         768×768
```

### **2. Confidence-Based Search Expansion**
- **High Confidence (>0.8)**: Normal search region
- **Medium Confidence (0.5-0.8)**: 1.5× expanded search
- **Low Confidence (<0.5)**: 2.5× expanded search (re-detection mode)

### **3. Quality-Aware Template Update**
- Template bank with exponential moving average
- Updates only on high-quality frames (confidence >0.85, low occlusion)
- Maintains initial template for reference

### **4. Multi-Scale Re-Detection**
- Pyramid search at [384, 512, 768] pixels when object is lost
- Automatic recovery from temporary occlusions
- Maintains long-term tracking stability

## Key Technical Features

- **DINOv3-Small Backbone**: Efficient frozen backbone for fast training
- **TrulyDynamicCenterPredictor**: Head that handles variable feature map sizes
- **RoPE Positional Embeddings**: Natural handling of different resolutions
- **Template Bank**: Robust against temporary quality degradation
- **Occlusion Detection**: Score map entropy-based quality assessment

## Original OSTrack

This project is based on the official implementation for the **ECCV 2022** paper [_Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework_](https://arxiv.org/abs/2203.11991).

[[Models](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing)][[Raw Results](https://drive.google.com/drive/folders/1TYU5flzZA1ap2SLdzlGRQDbObwMxCiaR?usp=sharing)][[Training logs](https://drive.google.com/drive/folders/1LUsGf9JRV0k-R3TA7UFBRlcic22M4uBp?usp=sharing)]

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=joint-feature-learning-and-relation-modeling)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=joint-feature-learning-and-relation-modeling)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-trackingnet&#41;]&#40;https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=joint-feature-learning-and-relation-modeling&#41;)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=joint-feature-learning-and-relation-modeling)
 -->
<p align="center">
  <img width="85%" src="https://github.com/botaoye/OSTrack/blob/main/assets/arch.png" alt="Framework"/>
</p>

## News
**[Dec. 12, 2022]**
- OSTrack is now available in [Modelscope](https://modelscope.cn/models/damo/cv_vitb_video-single-object-tracking_ostrack/summary), where you can run demo videos online and conveniently integrate OSTrack into your code.

**[Oct. 28, 2022]**
- :trophy: We are the winners of VOT-2022 STb(box GT) & RTb challenges.


## Highlights

### :star2: New One-stream Tracking Framework
OSTrack is a simple, neat, high-performance **one-stream tracking framework** for joint feature learning and relational modeling based on self-attention operators.
Without any additional temporal information, OSTrack achieves SOTA performance on multiple benchmarks. OSTrack can serve as a strong baseline for further research.

| Tracker     | GOT-10K (AO) | LaSOT (AUC) | TrackingNet (AUC) | UAV123(AUC) |
|:-----------:|:------------:|:-----------:|:-----------------:|:-----------:|
| OSTrack-384 | 73.7         | 71.1        | 83.9              | 70.7        |
| OSTrack-256 | 71.0         | 69.1        | 83.1              | 68.3        |


### :star2: Fast Training
OSTrack-256 can be trained in ~24 hours with 4*V100 (16GB of memory per GPU), which is much faster than recent SOTA transformer-based trackers. The fast training speed comes from:

1. While previous Siamese-style trackers required separate feeding of the template and search region into the backbone at each iteration of training, OSTrack directly combines the template and search region. The tight and highly parallelized structure results in improved training and inference speed.
  
2. The proposed early candidate elimination (ECE) module significantly reduces memory and time consumption.
  
3. Pretrained Transformer weights enable faster convergence.

### :star2: Good performance-speed trade-off

[//]: # (![speed_vs_performance]&#40;https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png&#41;)
<p align="center">
  <img width="70%" src="https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png" alt="speed_vs_performance"/>
</p>

## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training

### Prerequisites

Download pre-trained [DINOv3 ViT-Small-16 weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) and put it under `$PROJECT_ROOT$/pretrained_models/`:

```bash
mkdir -p pretrained_models
cd pretrained_models
# Download DINOv3-Small pretrained weights
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
mv dinov2_vits14_pretrain.pth dinov3_vits16_pretrain.pth
cd ..
```

### Training DOSTrack

```bash
# Train with adaptive sizing (recommended)
python tracking/train.py \
  --script dostrack \
  --config dostrack_adaptive \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_wandb 1
```

**Training Features:**
- **Frozen Backbone**: DINOv3-Small stays frozen, only head is trained
- **Fast Convergence**: ~12-15 hours on 4× V100 GPUs (vs. 24h+ for full training)
- **Batch Size**: 64 (larger than traditional due to frozen backbone)
- **Adaptive Training**: Supports variable template/search sizes during training

**Configuration Options** (`experiments/dostrack/dostrack_adaptive.yaml`):
```yaml
# Enable/disable adaptive features
DATA.ADAPTIVE.ENABLED: True
TEST.ADAPTIVE.ENABLED: True

# Adjust sizing thresholds
DATA.ADAPTIVE.SIZE_THRESHOLDS: [50, 150, 300]
DATA.ADAPTIVE.TEMPLATE_SIZES: [64, 128, 224, 384]

# Template update parameters
DATA.ADAPTIVE.TEMPLATE_UPDATE.CONFIDENCE_THRESHOLD: 0.85
DATA.ADAPTIVE.TEMPLATE_UPDATE.UPDATE_INTERVAL: 5
```


## Evaluation

### Setup

Put the trained weights in: `$PROJECT_ROOT$/output/checkpoints/train/dostrack/dostrack_adaptive/DOSTrack_ep0300.pth.tar`

Update dataset paths in `lib/test/evaluation/local.py`

### Testing with Adaptive Tracking

```bash
# LaSOT benchmark
python tracking/test.py dostrack dostrack_adaptive \
  --dataset lasot \
  --threads 16 \
  --num_gpus 4

# GOT10K-test
python tracking/test.py dostrack dostrack_adaptive \
  --dataset got10k_test \
  --threads 16 \
  --num_gpus 4

# TrackingNet
python tracking/test.py dostrack dostrack_adaptive \
  --dataset trackingnet \
  --threads 16 \
  --num_gpus 4
```

### Debug Mode (Visualize Adaptive Behavior)

```bash
python tracking/test.py dostrack dostrack_adaptive \
  --dataset vot22 \
  --threads 1 \
  --num_gpus 1 \
  --debug 1
```

**Debug output shows:**
- Current template and search sizes
- Confidence scores
- Template update events
- Size adjustments

**Debug visualizations saved to:** `debug/0001.jpg`, `debug/0002.jpg`, etc.

### Adaptive Features in Testing

When `TEST.ADAPTIVE.ENABLED: True`:
- ✅ **Dynamic sizing** based on object size
- ✅ **Confidence-based expansion** of search region
- ✅ **Template updates** with quality assessment
- ✅ **Multi-scale re-detection** when object is lost
- ✅ **Automatic size adjustment** every 5 frames

Example output:
```
[DOSTrack] Adaptive sizing initialized:
  Object size: 45.3x67.8
  Template size: 64x64
  Search size: 192x192
  Search factor: 3.00

[DOSTrack] Template updated at frame 15, conf=0.891
[DOSTrack] Adjusting sizes:
  Template: 64 -> 128
  Search: 192 -> 256
[DOSTrack] Low confidence (0.287), lost count: 1
[DOSTrack] Re-detected at scale 1
```

## Visualization or Debug
[Visdom](https://github.com/fossasia/visdom) is used for visualization.
1. Alive visdom in the server by running `visdom`:

2. Simply set `--debug 1` during inference for visualization, e.g.:
```bash
# DOSTrack visualization
python tracking/test.py dostrack dinov3_vitb16_no_ce --dataset vot22 --threads 1 --num_gpus 1 --debug 1

# Original OSTrack visualization
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser (remember to change the IP address and port according to the actual situation).

4. Then you can visualize the tracking process and feature maps.

![ECE_vis](https://github.com/botaoye/OSTrack/blob/main/assets/vis.png)


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```bash
# Profiling DOSTrack with DINOv3
python tracking/profile_model.py --script dostrack --config dinov3_vitb16_no_ce

# Profiling original OSTrack models
python tracking/profile_model.py --script dostrack --config vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script dostrack --config vitb_384_mae_ce_32x4_ep300
```


## Architecture Overview

DOSTrack extends the OSTrack framework with DINOv3 integration:

- **Backbone**: DINOv3 Vision Transformers (ViT-S/B/L/G) with RoPE positional embeddings
- **Feature Fusion**: One-stream architecture for joint template-search processing
- **Head**: Center-based or corner-based prediction heads
- **Training**: Supports frozen backbone, LoRA fine-tuning, or full fine-tuning

## Acknowledgments

* Thanks to the [OSTrack](https://github.com/botaoye/OSTrack) team for the excellent tracking framework
* Thanks to Meta AI for [DINOv3](https://github.com/facebookresearch/dinov3) pre-trained models
* Thanks for the [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) libraries
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo

## Citation

If you use DOSTrack in your research, please cite the original OSTrack paper:

```bibtex
@inproceedings{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2022}
}
```

And consider citing DINOv3:

```bibtex
@article{oquab2023dinov3,
  title={DINOv3: Scaling Self-Supervised Learning with Vision Transformers},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

---

# DOSTrack - DINO-based Object Tracking
