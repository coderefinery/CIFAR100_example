# CIFAR100_example

An example ML pipeline that trains a model on the CIFAR100 dataset


## How to run

1. Install the required packages

```bash
pip install -r requirements.txt
```

2. Run the training script

```bash
python cifar_detector.py
```

When running for the first time, the script will download the CIFAR100 dataset.
If a cached dataset is found, it is used instead.
