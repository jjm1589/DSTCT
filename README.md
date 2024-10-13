# DSTCT

## Early Accept in MICCAI 2024

The PSFHS Challenge of MICCAI 2023 and the IUGC Challenge of MICCAI 2024 are available at https://ps-fh-aop-2023.grand-challenge.org/ and https://codalab.lisn.upsaclay.fr/competitions/18413, respectively.

## Framework:

![Alt](framework.png)

## Usage

1. Clone the repo.;

```
https://github.com/jjm1589/DSTCT.git
```

2. Put the data in './data/FHPS/';

```
run fhps_data_processing.py and fhps_data_processing2.py
```

3. Train the model;

```
cd code
run ./train_fhps_unet_vit_semi_seg.sh
```

4. Test the model;

```
cd code
run ./test_fhps_unet_vit_semi_seg.sh
```
