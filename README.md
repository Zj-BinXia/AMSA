# AMSA
This project is the official implementation of 'Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-based Super-Resolution', AAAI22

The code framework is mainly modified from [BasicSR](https://github.com/xinntao/BasicSR) and [MMSR](https://github.com/open-mmlab/mmediting) (Now reorganized as MMEditing). Please refer to the original repo for more usage and documents.

## Dependencies and Installation

- Python == 3.8
- PyTorch == 1.4
- CUDA 10.0 
- GCC 5.4.0

1. Install Dependencies

   ```bash
   cd AMSA
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
