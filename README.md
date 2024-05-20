# WATS-DA
## Wildlife2024 Dataset

![image](img/Wildlife2024.png)
### Download
Download Wildlife2024, link:https://pan.quark.cn/s/f9aec26b21a5 code：4whM
The training set for the dataset Wildlife2024 is available for download on [Drive](https://pan.quark.cn/s/f9aec26b21a5),code：4whM, and the test set is [WATB](https://w-1995.github.io/EvaluationandResult.html).
## Trackers


| Trackers | Description | Language | Trackers | Description | Language |
| :---------- | :-----------: | :---------: | :---------- | :-----------: | :---------: |
| SMAT | WACV2024 | Python | CNNInMo| IJCAI2022 | Python |
| GRM | CVPR2023 | Python | MRCF | TIE2022 | Matlab |
| SGDViT| ICRA2023 | Python | TransT | CVPR2021 | Python |
| TCTrack++| TPAMI2023 | Python | STMTrack | CVPR2021 | Python |
| ETTrack | WACV2023 | Python | SiamGAT | CVPR2021 | Python |
| CTTrack | AAAI2023 | Python | HiFT| ICCV2021 | Python |
| SiamTPN | WACV2022 | Python | SiamCAR | CVPR2020 | Python |
| SiamRBO | CVPR2022 | Python | SiamBAN | CVPR2020 | Python |

## evaluation

| Trackers | Precision | Normalized Precision | Success | Trackers | Precision | Normalized Precision | Success |
| :---------- | :-----------: | :-----------: | :-----------: | :---------- | :-----------: | :-----------: | :-----------: |
| SMAT | 0.556 | 0.554 | 0.487 | CNNInMo | 0.527 | 0.518 | 0.455 |
| GRM | 0.549 | 0.539 | 0.491 | MRCF | 0.368 | 0.340 | 0.303 |
| SGDViT | 0.482 | 0.486 | 0.427 | TransT | 0.550 | 0.547 | 0.492 |
| TCTrack++ | 0.509 | 0.502 | 0.437 | STMTrack | 0.556| 0.553 | 0.486 |
| ETTrack | 0.506 | 0.512 | 0.454 | SiamGAT | 0.344 | 0.333 | 0.322 |
| CTTrack | 0.485 | 0.488 | 0.432 | HiFT | 0.288 | 0.270 | 0.284 |
| SiamTPN | 0.462 | 0.471 | 0.417 | SiamCAR | 0.516 | 0.501 | 0.423 |
| SiamRBO | 0.548 | 0.539 | 0.464 | SiamBAN | 0.538 | 0.531 | 0.455 |
| WATS-DA(CAR) | 0.564 | 0.553 | 0.493 | WATS-DA(GAT) | 0.443 | 0.443 | 0.418 |		
|WATS-DA(RBO) |	0.552 |	0.538 |	0.495 |WATS-DA(BAN) | 0.535 | 0.519 | 0.462 |

## test

Download pre-trained models for [ar](https://drive.google.com/drive/folders/1_dPapMvHy1iewJ2MmyWXj3BB-aIlHNLA?usp=sharing) and [re-detection](https://drive.google.com/drive/folders/1yIeq0CCi-JFDECafOA9fPkZSa_v9ILPL?usp=sharing) before testing.Pre-trained models for testing are acquired：

| Trackers | model | 
| :---------- | :-----------: | 
| WATS-DA(CAR) | [car](https://drive.google.com/file/d/17ZX_PwSKtrQGjlVsRaw-VXacc4XIe4yd/view?usp=sharing) | 
| WATS-DA(GAT) | [gat](https://drive.google.com/file/d/1ug3VJCSOH0yf6Rc38A-GdPXWU-9FcQoL/view?usp=sharing) | 
| WATS-DA(RBO) | [rbo](https://drive.google.com/file/d/1i_eoOnHCUb6AF6K76PDd2UuygjUYQ0mp/view?usp=sharing) |
| WATS-DA(BAN) | [ban](https://drive.google.com/file/d/1XNugJ2i0Wqi_31wmXjy8H_5wfsrb5_lA/view?usp=sharing) |

## train

Please download the pre-training model for each baseline tracker prior to training.

| Trackers | model | 
| :---------- | :-----------: | 
| siamcar | [herf](https://drive.google.com/file/d/15GXHlNz1OzRnIT4mFyziEvquf3cUVppQ/view?usp=sharing) | 
| siamgat | [herf](https://drive.google.com/file/d/1a7P3BlCwFUFIdtcs-1MOXUF5gRuSzt3T/view?usp=sharing) | 
| siamrbo | [herf](https://drive.google.com/file/d/1OdBbVLmbRCef3uq9D40JfDy99uAWB3XW/view?usp=sharing) |
| siamban | [herf](https://drive.google.com/file/d/1ScO4INZDF2iwolsz-eAslI3h4wmhkbuv/view?usp=sharing) |
