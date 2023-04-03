



# Performance on multimodal classificaition task

All the files used for multimodal classificaition task are contained in the classification floder. And its structures are introduced below.

- Configuration: containing the config file for 


## Prepair dataset
- Download
  - [dataset link](https://github.com/liuajian/Face-Anti-spoofing-Datasets)
- Create soft link to data folder
    - ln -s data_path path_to_MMANet/data/dataset_name
    - e.g., ln -s /home/ssd/CASIA-SURF /home/bbb/shicaiwei/MMANet/data/CASIA-SURF



## Inference
- Download Pretrained model from following links.
  - Pretrained multimodal model with complete data for [SURF](https://drive.google.com/drive/folders/1PxuXC2GfOsOUJl5HLPTP3HVF6wM1pcxC) dataset
  - Pretrained multimodal model with complete data for [CeFA](https://drive.google.com/drive/folders/18aTwbnv8ne29tYtyJcPpDqzG2nfxaWBU) dataset

- create folder and move the pretrained model in it
```bash
  cd classification
  mkdir output
  cd output
  mkdir models
  mv path_to_model/*.pth ./models
```

- testing with pretrained models
```bash
cd classification/test 
python surf_mmanet.py 0 0 0 0 0 0
python cefa_mmanet.py 0 0 0 0 0 
```


## Training From Scratch 

### Get multimodal model with complete data
```bash
cd classification/src
bash surf_multi.sh   #model for CASIA-SURF dataset
bash cefa_multi.sh   #model for CeFA dataset
```

### Test multimodal model with complete data
```bash
cd classification/test 
python baseline_multi_test.py 0 0 0 0 0
```
- Here the parameters are set as 0 since they have been set in the python file


### Get MMANet model for incomplete data
```angular2html
cd classification/src
bash surf_mmanet.sh   #model for CASIA-SURF dataset
bash cefa_mmanet.sh  #model for CeFA dataset
```
### Test multimodal model with incomplete data
```bash
cd classification/test 
python surf_mmanet.py 0 0 0 0 0 0
python cefa_mmanet.py 0 0 0 0 0 
```

