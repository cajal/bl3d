""" Some miscellaneous scripts """


# Training and evaluation
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=True # faster: 30secs vs 96 secs per epoch (without it)

from bl3d import train
train.TrainedModel().populate('model_hash LIKE "fcn%%"', reserve_jobs=True)
#train.TrainedModel().populate(reserve_jobs=True)

from bl3d import evaluate
evaluate.SegmentationMetrics().populate(reserve_jobs=True)



# Getting evaluation results
from bl3d import params, train, evaluate

params_rel = params.TrainingParams() & {'lr_schedule': 'none', 'positive_weight': 1}
thashes = params_rel.fetch('training_hash', order_by=['learning_rate', 'weight_decay'])
train_losses = {}
val_losses = {}
best_val_losses = {}
best_train_losses = {}
best_epochs = {}
best_thresholds = {}
best_IOUs = {}
model = 'model_hash LIKE "fcn_9%%"'
set_ = 'val'
for th in thashes:
    train_losses[th] = (train.TrainedModel() & {'training_hash': th} & model).fetch('train_loss')
    val_losses[th] = (train.TrainedModel() & {'training_hash': th} & model).fetch('val_loss')
    best_val_losses[th] = (train.TrainedModel() & {'training_hash': th} & model).fetch('best_val_loss')
    best_train_losses[th] = (train.TrainedModel() & {'training_hash': th} & model).fetch('best_train_loss')
    best_epochs[th] = (train.TrainedModel() & {'training_hash': th} & model).fetch('best_epoch')
    best_thresholds[th] = (evaluate.SegmentationMetrics() & {'training_hash': th, 'set': set_} & model).fetch('best_threshold')
    best_IOUs[th] = (evaluate.SegmentationMetrics() & {'training_hash': th, 'set': set_} & model).fetch('best_iou')

## Plot losses
import matplotlib.pyplot as plt
plt.ioff()
for th in thashes:
    lr, wd = (params.TrainingParams & {'training_hash': th}).fetch1('learning_rate', 'weight_decay')
    print('th', th)
    print('lr', lr)
    print('lambda', wd)
    #for t in train_losses[th]: plt.plot(t)
    #for t in val_losses[th]: plt.plot(t)
    plt.plot(train_losses[th].mean(axis=0))
    plt.ylim([0.1, 0.6])
    plt.show()

## Report results
for th in thashes:
     lr, wd = (params.TrainingParams & {'training_hash': th}).fetch1('learning_rate', 'weight_decay')
     print('th', th)
     print('lr', lr)
     print('lambda', wd)
     print('best epoch (mean)', best_epochs[th].mean())
     print('best_val_loss (mean)', best_val_losses[th].mean())
     print('best_train_loss (mean)', best_train_losses[th].mean())
     print('best_threshold (mean)', best_thresholds[th].mean())
     print('best_iou (mean)', best_IOUs[th].mean())
     print('')