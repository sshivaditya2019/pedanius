import os
import random
import numpy as np

import torch



model_version="cnn"
subset_percent= 1.0
augmentation= "yes"
annotations_file="/home/sshivaditya/Projects/pedanius/data/train.csv"
img_dir="/home/sshivaditya/Projects/pedanius/data/train_images"
teacher="pre-trained"
alpha=0
train_size=0.8
test_size=0.2
temperature=1
learning_rate=1e-3
batch_size=128
num_epochs=30
dropout_rate=0.5
num_channels=32
save_summary_steps=100
num_workers=4
fold_num= 5
seed= 719
img_size= 256
epochs= 10
train_bs= 16
valid_bs= 32
T_0= 10
lr= 1e-4
min_lr= 1e-6
weight_decay=1e-6
accum_iter= 2
verbose_step= 1




# INPUT_PATH            = "./input"  # PC
INPUT_PATH            = "/home/sshivaditya/Projects/cassava_model/" # Kaggle
# INPUT_PATH            = "."        # Colab
GENERATED_FILES_PATH  = "/home/sshivaditya/Projects/CKD/generated/"
DATASET_PATH          = os.path.join(INPUT_PATH, "data/")
TRAIN_IMAGES_DIR      = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR       = os.path.join(DATASET_PATH, "test_images")
TRAIN                 = os.path.join(DATASET_PATH, "train.csv")
TEST                  = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS           = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
WEIGHTS_PATH          = "/home/sshivaditya/Projects/CKD/generated/weights/" # For PC and Kaggle
#WEIGHTS_PATH          = "/content/drive/My Drive/weights" # For Colab
USE_GPU               = True # [True, False]
USE_TPU               = False # [True, False]
GPUS                  = 1
TPUS                  = 8 # Basically TPU Nodes
PARALLEL_FOLD_TRAIN   = False # [True, False]
SEED                  = 719
FOLDS                 = 5
MIXED_PRECISION_TRAIN = True # [True, False]
DROP_LAST             = True # [True, False]
DO_FREEZE_BATCH_NORM  = True # [True, False]
FREEZE_BN_EPOCHS      = 5


ONE_HOT_LABEL         = False
DO_DEPTH_MASKING      = False
DO_FMIX               = False
FMIX_PROBABILITY      = 0.5
DO_CUTMIX             = False
CUTMIX_PROBABILITY    = 0.5


H                     = 256 # [224, 384, 512]
W                     = 256 # [224, 384, 512]
PATH_TO_TEACHER       = "/home/sshivaditya/Projects/CKD/saves/SEResNeXt50_32x4d_BH_fold_3_14"
TEACHER_NAME          = "SEResNeXt50_32x4d_BH"
OPTIMIZER             = "RAdam"  # [Adam, AdamW, RAdam, AdaBelief, RangerAdaBelief]
SCHEDULER             = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
SCHEDULER_WARMUP      = True # [True, False]
WARMUP_EPOCHS         = 1 if SCHEDULER_WARMUP else 0
WARMUP_FACTOR         = 7 if SCHEDULER_WARMUP else 1
TRAIN_CRITERION       = "BiTemperedLogisticLoss" # [BiTemperedLogisticLoss, LabelSmoothingCrossEntropy, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
VALID_CRITERION       = "SoftmaxCrossEntropy" # [BiTemperedLogisticLoss, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15
SCHEDULER_BATCH_STEP  = True # [True, False]

N_CLASSES             = 5

TRAIN_BATCH_SIZE      = 16
VALID_BATCH_SIZE      = 8
ACCUMULATE_ITERATION  = 2

NET                   = "cnn" # [SEResNeXt50_32x4d_BH, ResNeXt50_32x4d_BH, ViTBase16_BH, ViTBase16, ViTLarge16
                                             #  resnext50_32x4d, seresnext50_32x4d, tf_efficientnet_b4_ns, ['vit_base_patch16_224', 'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_base_resnet26d_224', 'vit_base_resnet50d_224', 'vit_huge_patch16_224', 'vit_huge_patch32_384', 'vit_large_patch16_224', 'vit_large_patch16_384', 'vit_large_patch32_384', 'vit_small_patch16_224', 'vit_small_resnet26d_224', 'vit_small_resnet50d_s3_224']

PRETRAINED            = True
LEARNING_VERBOSE      = True
VERBOSE_STEP          = 100

USE_SUBSET            = False
SUBSET_SIZE           = TRAIN_BATCH_SIZE * 1
CPU_WORKERS           = 4

TRAIN_BATCH_SIZE    //= ACCUMULATE_ITERATION
VALID_BATCH_SIZE    //= ACCUMULATE_ITERATION
if not PARALLEL_FOLD_TRAIN:
    if USE_TPU:
        TRAIN_BATCH_SIZE //= TPUS
        VALID_BATCH_SIZE //= TPUS
    elif USE_GPU:
        TRAIN_BATCH_SIZE //= GPUS
        VALID_BATCH_SIZE //= GPUS


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)