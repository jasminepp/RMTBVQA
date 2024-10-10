
### RMT-BVQA: Recurrent Memory Transformer-based Blind Video Quality Assessment for Enhanced Video Content [arXiV](https://arxiv.org/abs/2405.08621)

This paper is accepted by **ECCV2024 Advanced in image manipulation (AIM) Workshop**.

---
### Contribution

   Developed a **large training database** with enhanced video content.
   Designed a **Recurrent Memory Vision Transformer (RMViT)** module.
   Proposed a **new contrastive learning training strategy** for blind VQA.

---
### Usage
#### Training
First, download the training dataset and put it in the `./training_data/` directory and put its corresponding label file in the `./csv_files/` directory.

Finally, use pretrained `spatial feature extractor` to extract frame-level features of the training dataset.
```
python spatial_feature_extract.py
```
Finally, train the RMT-BVQA model by run the following command.
```
python train.py --num_memory 12 --size_seg 4 --lr 0.00025 --batch_size 256
```
---
#### Testing
There are three steps for testing.

1. Extract frame-level features of the testing dataset [VDPVE](https://github.com/YixuanGao98/VDPVE-VQA-Dataset-for-Perceptual-Video-Enhancement).
```
python spatial_feature_extract.py
```

2. Extract video features by trained RMT-BVQA model.
```
python demo_feat.py

```
3. Train a regressor and predict the quality score (5-Fold Cross Validation).

```
python train_regressor.py
python test_regressor.py
```
---
#### Environment
> python 3.8
> pytorch 1.10.0
> cuda 11.3


### Citation
Please consider citing our work if you find that it is useful. Much appreciated!
>@misc{peng2024rmtbvqa,
      title={RMT-BVQA: Recurrent Memory Transformer-based Blind Video Quality Assessment for Enhanced Video Content}, 
      author={Tianhao Peng and Chen Feng and Duolikun Danier and Fan Zhang and Benoit Vallade and Alex Mackin and David Bull},
      year={2024},
      eprint={2405.08621},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2405.08621}, 
} 