# GSA
Code for the paper "Improving Adversarial Transferability with Ghost Samples", ECAI 2023.

## Prepare the Dataset
We use the ImageNet-compatible dataset provided by the NIPS 2017 adversarial competition. 

Download the dataset following the instruction in their [github repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

## Attack
Generate GSA adversarial examples:

```
python attack.py --GSA --aug_num 15 --loss_function MaxLogit --src_model resnet_50
```
We also provide a script for generating adversarial examples using various transfer-based attack methods.
```
sh attack.sh
```

## Evaluation
```
python eval.py --img_dir your_adversarial_example_path.npy
```

