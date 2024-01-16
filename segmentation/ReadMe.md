

## Dataset
### Download
- NYUV2
```bash
  cd segmentation/src/datasets/nyuv2
  python prepare_dataset.py path_to_save_data
  
  for example: python prepare_dataset.py /home/data/shicaiwei/NYUV2
```
- CityScapes
```bash
  cd segmentation/src/datasets/cityscapes 
  python prepare_dataset.py path_to_save_data path_to_save_processing_data
```

### Build soft link
```bash
   cd segmentation
   mkdir data
   ln -s path_to_nyuv2_dataset ./data/NYUV2
   ln -s path_to_cityscape_datset ./data/CityScape
```

[//]: # (## Inference)

[//]: # (- Download Pretrained model from following links.)

[//]: # (    - Pretrained multimodal model with complete data for [NYUV2]&#40;https://drive.google.com/drive/folders/1PxuXC2GfOsOUJl5HLPTP3HVF6wM1pcxC&#41; dataset)

[//]: # (    - Pretrained multimodal model with complete data for [CityScape]&#40;https://drive.google.com/drive/folders/18aTwbnv8ne29tYtyJcPpDqzG2nfxaWBU&#41; dataset)

[//]: # ()
[//]: # (- create folder and move the pretrained model in it)

[//]: # (```bash)

[//]: # (  cd segmentation/test)

[//]: # (  mkdir output)

[//]: # (  cd output)

[//]: # (  mkdir models)

[//]: # (  mv path_to_model/*.pth ./models)

[//]: # (```)

[//]: # ()
[//]: # (- testing with pretrained models)

[//]: # (```bash)

[//]: # (cd classification/test )

[//]: # (python surf_mmanet.py 0 0 0 0 0 0)

[//]: # (python cefa_mmanet.py 0 0 0 0 0 )

[//]: # (```)

## Inference

### Download pretrain model
- NYUV2: [baiduyun](https://pan.baidu.com/s/1UY-fzsrGxTR4-lJ6zAlyFA) itas
  - full folder contains the baseline model trained with complete multimodal data
  - mad_mar folder contains the models trained with mad and mar strategies
- Cityscape
  - [baiduyun](https://pan.baidu.com/s/1DAlAuUlK9ROegmFlk3B2cw?pwd=0n3p) 0n3p

- modify the path to pretrained model in 'test/test_missing.py' (Line 99 for NYUV2 and Line 109 for Cityscape)
- run code 
    ```bash
      cd test
      bash test_nyuv2.sh
      bash test_cityscape.sh
      
    ```


## Training From Scratch

### Train multimodal model
```bash
    bash train_nyuv2_full.sh 
    bash train_train_cityscape_full.sh
```


### Train Get MMANet model for incomplete data model
```bash
    bash train_nyuv2_mmanet.sh 
    bash train_train_cityscape_mmanet.sh
```




