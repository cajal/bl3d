""" Some miscellaneous scripts """

#####################################################################################
# Training and evaluation
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=True # faster: 30secs vs 96 secs per epoch (without it)

from bl3d import train
train.TrainedModel().populate('model_hash LIKE "fcn%%"', reserve_jobs=True)
#train.TrainedModel().populate(reserve_jobs=True)

from bl3d import evaluate
evaluate.SegmentationMetrics().populate(reserve_jobs=True)


####################################################################################
# Getting evaluation results
from bl3d import params, train, evaluate

params_rel = params.TrainingParams() & {'lr_schedule': 'none', 'positive_weight': 4}
thashes = params_rel.fetch('training_hash', order_by=['learning_rate', 'weight_decay'])
train_losses = {}
val_losses = {}
best_val_losses = {}
best_train_losses = {}
best_epochs = {}
best_thresholds = {}
best_IOUs = {}
model = 'model_hash LIKE "fcn_4%%"'
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
    for t in train_losses[th]: plt.plot(t)
    #for t in val_losses[th]: plt.plot(t)
    #plt.plot(train_losses[th].mean(axis=0))
    plt.ylim([0.1, 1.0])
    plt.show()

# Report results (in matrix form)
wds = [0, 0.00001, 0.001, 0.1, 10]
lrs = [0.0001, 0.001, 0.01, 0.1]
for metric, decimals in [(best_val_losses, 3), (best_epochs, 0), (best_thresholds, 2), (best_IOUs, 3)]:
    print('##')
    print('| Lambda/LR\t| 0.0001| 0.001\t| 0.01\t| 0.1\t|')
    for i, wd in enumerate(wds):
        res = [round(metric[th].mean(), decimals) for th in thashes[i: len(wds) * len(lrs): len(wds)]]
        print('| {}\t\t| {}\t|'.format(wd, '\t| '.join(map(str, res))))
    print('')

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



####################################################################################
# Running net on a new stack (from bl3d schema or from stack pipeline)
import matplotlib.pyplot as plt
from bl3d.evaluate import *
import datajoint as dj

rel = dj.U('model_hash', 'training_hash', 'set').aggr(SegmentationMetrics() & 'model_hash LIKE "fcn_9%%"' & {'set': 'val'}, avg='AVG(best_iou)')
key = (rel & 'avg > 0.45').fetch1('KEY')
key['val_id'] = 2 # select one of the 8 trained networks

net = train.TrainedModel.load_model(key)

torch.cuda.set_device(7)
net.cuda()

examples = [key['val_id']]
dataset = datasets.SegmentationDataset(examples, transforms.ContrastNorm())
dataloader = DataLoader(dataset, num_workers=2, pin_memory=True)

image, label = iter(dataloader).__next__()
with torch.no_grad():
     output = forward_on_big_input(net, Variable(image.cuda()))
torch.save(output, '/mnt/lab/users/ecobost/output_example2.torch')

### same in a meso stack
stack = dj.create_virtual_module('meso', 'pipeline_stack')
key = {'animal_id': 17977, 'session': 5, 'stack_idx': 10, 'channel': 1}
slices = (stack.CorrectedStack.Slice() & key).fetch('slice', order_by='islice')
stack = np.stack(slices)
image2 = torch.from_numpy(stack).view(1, 1, 700, 1364, 1399)
with torch.no_grad():
     output2 = forward_on_big_input(net, Variable(image2.cuda()))
torch.save(output2, '/mnt/lab/users/ecobost/output_meso.torch')

## In other computer
import torch
import torch.nn.functional as F
output = torch.load('/mnt/lab/users/ecobost/output_example2.torch')
prediction = F.softmax(output, dim=1)
prediction.shape # [1, 2, 199, 498, 497]

from bl3d import data
ex2 = (data.Stack.Volume() & {'example_id': 2}).fetch1('volume')
lbl2 = (data.Stack.Label() & {'example_id': 2}).fetch1('label')
pred = prediction[0, 1, :, :, :].numpy()

import matplotlib.pyplot as plt
from matplotlib import animation
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(19, 9))
im = axes[0].imshow(ex2[100])
im2 = axes[1].imshow(pred[100], vmin=0, vmax=1)
def update_img(i): im.set_data(lbl2[i]); im2.set_data(pred[i]); print(i)
video = animation.FuncAnimation(fig, update_img, 199, interval=150, repeat_delay=1000)
fig.tight_layout()
#video.save('/data/pipeline/axonal_threshold_1_4.mp4', dpi=250)
fig.show()

import datajoint as dj
stack = dj.create_virtual_module('meso', 'pipeline_stack')
key = {'animal_id': 17977, 'session': 5, 'stack_idx': 10, 'channel': 1}
slices = (stack.CorrectedStack.Slice() & key).fetch('slice', order_by='islice')
stack = np.stack(slices)
output2 = torch.load('/mnt/lab/users/ecobost/output_meso.torch')
prediction2 = F.softmax(output2, dim=1)
prediction2.shape # [1, 2, 199, 498, 497]
pred2 = prediction2[0, 1, :, :, :].numpy()

import matplotlib.pyplot as plt
from matplotlib import animation
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(19, 9))
im = axes[0].imshow(stack[100])
im2 = axes[1].imshow(pred2[100], vmin=0, vmax=1)
def update_img(i): im.set_data(stack[i]); im2.set_data(pred2[i]); print(i)
video = animation.FuncAnimation(fig, update_img, 700, interval=150, repeat_delay=1000)
fig.tight_layout()
#video.save('/data/pipeline/axonal_threshold_1_4.mp4', dpi=250)
fig.show()
