# HuBMAP-HPA---Hacking-the-Human-Body
https://www.kaggle.com/competitions/hubmap-organ-segmentation

## Objective:

Hubmap + HPA Hacking the Human Body was a competition launched to segregate or segment FTU (Functional Tissue Units) among cells. 

In the average kidney, there are over 1 million glomeruli FTUs. While there are existing cell and FTU segmentation methods, the objective of the competition to push the boundaries by building algorithms that generalize across different organs and are robust across different dataset differences.

## Problem Statement: 

- We are provided with a dataset consisting of tiff format train and test images , annotation files in json format. And other details like RLE (Run length encoding) in a csv. 
- Our aim is to put all together and build a model that could segment and bring out FTUs leaving out other organs.
- The competition also has some hidden dataset which will also be added upon in evaluation.

Size after split of data: Train= 280 Images, Valid = 42 Images, Test = 29 Images

## BEST SUBMISSION 

MODEL : SegFormer b4 finetuned ADE 512-512

Submissions Scores:

    - PRIVATE : 0.68734, 
    
    - PUBLIC: 0.68266,
    
    RANK: 410.

## List of Experiments

### Segmentation models in pytorch

#### Experiments with models :

| Decoder |  Encoder  | Train  | Valid | Test    |  Private |  Public  |
| ----- |  -----  | -----  | ----    | ----    |  ------- |  ------  |
| Constant Decoder | Change Encoders |   |  |   |  |  |
| Unet |  MobileNet V2  | 0.721  | 0.54 |  0.55  |  0.47132 | 0.53429  | 
|  | Timm-Res2next50 |  0.658  | 0.566  |  0.587  |  0.53091 | 0.5748  |
|  | VGG19_bn | 0.856  | 0.489  |  0.553  |  0.55638 | 0.62092  |
|  | Efficient Net b4 |  0.774  | 0.451  | 0.476  | 0.43136  | 0.44308 |
| |  |    |  |   |  |  |
| DEEPLABSV3++ | Mobilenet V2 |  0.654  | 0.492  |  0.560  | 0.35335  | 0.42664  |
|  | Efficientnet-b7 |  0.694  | 0.534  |  0.630  | 0.33126  | 0.42143  |
| |  |    |  |   |  |  |
| Change Decoder | Constant Encoders |   |  |   |  |  |
| Unet | VGG19_bn |  0.843  | 0.466  |  0.510  | 0.54539  | 0.61645  |
| Linknet |  |  0.871  | 0.427  |  0.465  | 0.5477  | 0.61182  |
| UnetPlusPlus |  |  0.930  | 0.700  |  0.687  | 0.48033  | 0.54471  |
| |  |    |  |   |  |  |


Best Combo: UNet, Vgg-19bn

#### Experiment with Hyperparameters (LR_Schedulers, Optimizers, Activation functions)

| Parameter tuned | Trial | Train  | Valid | Test    |  Private |  Public  |
| ----- | ----- |  -----  | ----  | ----    |  ------- |  ------  |
| LR_Scheduler | StepLR |  0.639  | 0.583  |  0.594  |  0.49772 | 0.49372  | 
| | LambdaLR |  0.953  | 0.696  |  0.702  | 0.49806  | 0.55773  | 
| | Multistep LR |  0.871  | 0.696  |  0.696  |  0.47839 |  0.53487 |
| | CosineAnnealingLR |  0.827  | 0.678  |  0.707  |  0.49847 |  0.54805 |
| | CosineWarmingRestartLR |  0.908  | 0.708  |  0.701  | 0.49187  |  0.54951 |
| | ExponentialLR |  0.639  | 0.583  |  0.594  |  0.49772 | 0.49372  |
| | LinearLR |  0.812  |  0.666 |  0.621  |  0.44134 |  0.51087 |
|  |  | |  |     |   |    |
| Optimizer | Adam | 0.856  | 0.489  |  0.553  |  0.55638 | 0.62092  |
| | AdamW |  0.854  | 0.651  |  0.685  |  0.48983 |  0.55120 | 
| | SGD |  0.951  | 0.685  |  0.666  |  0.46213 |  0.52275 |
|  |  | |  |     |   |    |
| Activation Fns | Tanh |  0.839  | 0.689  |  0.704  |  0.40839 |  0.35819 |
| | Sigmoid |  0.886  | 0.707  |  0.732  |  0.48568 |  0.55205 |  
| |  Identity | 0.856  | 0.489  |  0.553  |  0.55638 | 0.62092  |

Best Combo: Cosine Annealing LR, AdamW, Identity

#### Experiment with Loss Functions

| Loss Functions | Train  | Valid | Test    |  Private |  Public  |
| ----- |  -----  | ----  | ----    |  ------- |  ------  |
| BCE + lovasz_tversky_loss |  0.866  |  0.618 |  0.662  |  0.50363 |  0.55394 |
|  Focal + tversky_loss |  0.756  |  0.681 |  0.658  |  0.50496 |  0.56365 |
| BCE + Dice loss(For reference) |  0.856  |  0.489 |  0.553  |  0.55638 |  0.62092 |

Best Combo: BCE + Dice Loss

## Transformer Models

Initially experimented with different available models to fix upon the best performing one. This was done with less epochs(20-30) to facilitate trying out multiple models.

| Model |  Train  | Valid  | Test    |  Private |  Public  |
| ----- |  -----  | -----  | ----    |  ------- |  ------  |
| Mit_b5 |  0.814  | 0.719  |  0.706  |  0.55637 | 0.62621  | 
| Upernet-tiny |  0.826  | 0.696  |  0.667  |  0.45721 | 0.51646  |
| Upernet-small |  0.918  | 0.703  |  0.688  |  0.44187 | 0.49867  |
| Upernet - Large |  0.914  | 0.738  |  0.724  | 0.51249  | 0.55138  |
| Coat_lite_medium |  0.836  | 0.715  |  0.710  |  0.52425 |  0.61180  |
| Mask 2 Former-Swin_Large |  0.834  | 0.611  |  0.597  |  0.48598 | 0.55313  | 
| Mask 2 Former-Swin_Small |  0.862  | 0.692  |  0.72   |  0.52873 | 0.59574  |
| Segformer_semantic_segmentation |  0.878  |  0.719 | 0.696   | 0.60249  |  0.64491 |
| Segformer-b0-finetuned-ade-512-512 |  0.698  | 0.593  |  0.559  |  0.45025 |  0.43684 |
| Segformer-b4-finetuned-ade-512-512 |  0.826  |  0.729 |  0.699  |  0.63944 |  0.66985 |
| Segformer-b5-finetuned-ade-640-640 |  0.884  | 0.739  |  0.737  | 0.62117  |  0.67957 | 

Then went on with tuning of the tunable hyperparameters like optimizers, schedulers etc upon choosing some of the best performing models of the above experiments.

### Hyperparameters

Model = SegFormer b4 ADE finetuned 512-512

| Parameter tuned | Trial |  Train  | Valid  | Test    |  Private |  Public  |
| ----- | ----- |  -----  | -----  | ----    |  ------- |  ------  |
| Optimizers| Adam |  0.826  |  0.729 |  0.699  |  0.63944 |  0.66985 | 
|           | AdamW |  0.850  | 0.737  |  0.694  |  0.64349 |  0.68579 | 
|           | SGD |  0.743  | 0.706  |  0.685  |  0.62650 | 0.65561  | 
| LR_Schedulers | OnecycleLR |  0.396  | 0.523  |  0.468  | 0.50548  |  0.50869 | 
|               | CosineAnnealingLR |  0.871  | 0.736  |  0.688  | 0.66708  |  0.70351 | 

Here AdamW with Cosine Annealing LR scheduler combination yielded the best results. So continued with this combo of Segformer B4 with AdamW optimizer, CosineAnnealingLR learning rate scheduler for the upcoming experiments.

Best Combo

### Loss Functions 

| Trial |  Train  | Valid  | Test    |  Private |  Public  |
| ----- |  -----  | -----  | ----    |  ------- |  ------  |
| BCE + Lovasz Loss |  0.342  | 0.628  |  0.626  | 0.59852  |  0.60745 | 
| BCE + Dice Loss |  0.850  | 0.737  |  0.694  |  0.64349 |  0.68579 | 
| Jaccard + BCE + Focal + Tversky |  0.871  | 0.741  |  0.715  |  0.67329 | 0.70811  |

### Improvement Techniques 

Tried some methods to improve the performance of model such as Stochastic Weighted Averaging LR Scheduler, Grad scalar, Stain normalization etc. of which grad scaling proved better off in my submissions.

| Trial |  Train  | Valid  | Test    |  Private |  Public  |
| ----- |  -----  | -----  | ----    |  ------- |  ------  |
| SWA LR scheduler |  0.858  | 0.726  |  0.718  |  0.63196 |  0.67205 |  
| Grad scaling |  0.959  | 0.768  |  0.775  |  0.68734 | 0.68266  | 
| Stain Normalization(Vahadane) |    |   |    |  0.41738 | 0.38317  |
| Reinhard Color Normalization |    |   |    |  0.38543 | 0.36588  |

### Organ wise Training and prediction

This was a strategy used to identify the models inability to pick the FTUs in specific organ cause of bad detection of FTUs. 
Local CVs of organ wise training are as below:

| Organ |  Train  |  Test    |  Private |  Public  |
| ----- |  -----  |  ----    |   -----  |  ----    |
| Lung |  0.903  | 0.215  |     |      |
| Spleen |  0.922  | 0.745  |     |     |
| Prostate |  0.953  | 0.808  |     |      |
| Largeintestine |  0.964  | 0.877  |    |     |
| Kidney |  0.953  | 0.942  |     |      |
| Submission |    |   |  0.64869   |   0.6986   | 

Upon submissions it was understood that lung was the badly detected organ of all. Hence measures can be taken to give the model more information on lung to identify and segment it correctly.

### Augmentations

Made trials on these augmentations along with stratified Kfolds.
- Flips
- Huesaturation Value
- CLAHE
- Random brightness contrast

### Stratified K Fold Strategy

Model : Encoder = CoAT lite medium, Decoder = daformer conv 3x3
Image_size = 1024

| Fold | Train | Valid  |  Private |  Public  |
| ----- |  -----  | -----  |  ------- |  ------  |
| 0 |  0.899 | 0.780 | 0.54809 | 0.57976 |  
| 1 |  0.954 | 0.762 | 0.50194  | 0.51826  | 
| 2 |  0.971 | 0.743 | 0.57819 | 0.61537 | 
| 3 |  0.852 | 0.756 | 0.56477 | 0.58261 |
| 4 |  0.919 | 0.756 | 0.54118  | 0.57640  |
| |  Submission  | Ensemble of 5 folds  | 0.61851  | 0.65404  |


Model : Encoder = CoAT lite medium, Decoder = daformer conv 3x3
Image_size = 768

| Fold |  Train  | Valid  |  Private |  Public  |
| ----- |  -----  | -----  |  ------- |  ------  |
| 0 |  0.793  | 0.734  | 0.50676 |  0.56170 |  
| 1 |  0.819  | 0.648  | 0.55565  | 0.59312  | 
| 2 |  0.816  | 0.631  | 0.56824 | 0.59278 | 
| 3 |  0.796  | 0.696  | 0.54905 | 0.69923 | 
| |  Submission  | Ensemble of 4 folds  | 0.60639  | 0.64493  |

Model : SegFormer b4-ADE 512x512,
 Image_size = 768

| Fold |  Train  | Valid  |  Private |  Public  |
| ----- |  -----  | -----  |  ------- |  ------  |
| 0 |  0.814  | 0.766  | 0.57768 | 0.59394  |  
| 1 |  0.827  | 0.747  | 0.58218  |  0.61007 | 
| 2 |  0.784  | 0.752  | 0.68958 | 0.67833 | 
| |  Submission  | Ensemble of 3 folds  | 0.66319  | 0.67214  |
