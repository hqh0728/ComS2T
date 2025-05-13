

---

# ComS2T

This is the original PyTorch implementation of **ComS2T: A Complementary Spatiotemporal Learning System for Data-Adaptive Model Evolution**.

## Requirements

* Python 3
* Please install dependencies listed in `requirements.txt`

## Datasets

The following datasets are used in this project:

* **KnowAir (PM2.5 and Temperature)**: Available at [PM2.5GNN](https://github.com/YnnuSL/PM2.5-GNN)

  * Direct download link for the dataset: [BaiduYun](https://pan.baidu.com/s/1sdPFSR8Oq3XPrXMnXFqd1w)

* **METR-LA**: Available from the [DCRNN GitHub repository](https://github.com/liyaguang/DCRNN)

  * Direct download link: [https://github.com/liyaguang/DCRNN/tree/master/data](https://github.com/liyaguang/DCRNN/tree/master/data)

## Training

To train the model on the PM2.5 dataset:

```bash
sh ./run_pm2_5.sh
```

---
