# AMSA
This project is the official implementation of 'Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-based Super-Resolution', AAAI22

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.4.0](https://img.shields.io/badge/pytorch-1.4.0-green.svg?style=plastic)


## Dependencies and Installation

- Python >= 3.7
- PyTorch == 1.4
- CUDA 10.0 
- GCC 5.4.0


1. Install Dependencies

   ```bash
   cd C2-Matching
   conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch
   pip install mmcv==0.4.4
   pip install -r requirements.txt
   ```

1. Install MMSR and DCNv2

    ```bash
    python setup.py develop
    cd mmsr/models/archs/DCNv2
    python setup.py build develop
    ```


## Dataset Preparation

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.



