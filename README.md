# car_classification
Lab 02 of Computer Vision in Application course. I need to collect data and build transfer learning model to classify my own car dataset. I used EfficientNet as my base model to transfer learning on. I've followed these processes to do experiments:
- Data augmentation: rescale, shear, horizontal shift, vertical shift, zoom and rotation
- Feeze and pretrain: I added some layers on the top of EfficientNet for classifying on my own dataset such as Global Average Pooling, Dropout and FC (with softmax activation function) and feeze the pretrained weights of base model. Then, save my pretrained model.
- Fine-tune: Firstly, i loaded my pretrained model and then unfeeze some later layers of base model to retrain to obtain dataset-specific features on my own data and also increase the accuracy.

For optimizer, I chose Adam with catergorical cros-entropy as my loss function.

I have used the following stategies to have increase the efficient training:
- Learning rate schedule: I drop learning rate a half every 10 epochs
- Early stopping: I chose patience = 10 to wait and keep going until no changes to stop.

## Install requirements:
```
pip install -U -r requirements.txt
```

## Manual instruction
Get help:
```
python main.py -h
```

### Feeze and Pretrain

For example:
```
python main.py -ddir "../car_dataset" -af "../car_dataset/anotations.json" -e 50 \
                        -lr 1e-2 -kp 0.2 -b 32 -pl -dim 224 224 -bn B1
```

Arguments:
- ddir: data directory
- af: anotation file
- e: Number of epochs to run
- lr: learning rate
- kp: Keep prob
- b: batch size
- pl: plot and save learning curve
- dim: image dimension (height, width)
- bn: version of EfficientNet such as B0, B1, ..., B4

### Fine-tune

For example:
```
python main.py -ddir "../car_dataset" -af "../car_dataset/anotations.json" -e 50 \
                        -wdir "./models/model_2019_11_03-16_34_25.h5" \
                        -lr 2e-3 -kp 0.2 -b 32 -pl -dim 224 224 -bn B1 -fi \
                        -st "block7a_expand_conv (Conv2D)" -pt 10 \
```

Arguments:
- wdir: pretrained weights file
- fi: Enable fine-tuning stage
- st: start layer to train on EfficientNet
- pt: patience used for Early stopping

