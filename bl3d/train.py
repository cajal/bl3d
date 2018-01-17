""" Importing structural data from our pipeline. """
import datajoint as dj
import torch
import numpy as np
from torch import nn, optim
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.autograd import Variable

from bl3d import data
from bl3d import params
from bl3d import datasets
from bl3d import transforms


schema = dj.schema('ecobost_bl3d', locals())


def log(*messages):
    """ Simple logging function."""
    import time
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True)

def compute_loss(net, dataloader, criterion):
    """ Compute average loss over examples in a dataloader. """
    net.eval()

    loss = 0
    for volume, label in dataloader:
        volume = Variable(volume.cuda(), volatile=True)
        label = Variable(label.cuda(), volatile=True)
        output = net(volume) # 1 x num_classes x d x h x w
        loss += criterion(output.view(output.shape[1], -1).t(), label.view(-1)).data[0]
    loss /= len(dataloader)

    net.train()

    return loss


@schema
class Split(dj.Lookup):
    definition = """ # examples used for training and validation (no test set)
    val_id:                  int         # either example_id or animal_id
    ---
    num_examples:            int         # number of examples in this set
    train_examples:          longblob    # list of examples used for training
    val_examples:            longblob    # list of examples used in validation
    """
    @property
    def contents(self):
        contents = []
        example_ids = set(data.Stack().fetch('example_id'))
        num_examples = len(example_ids)
        for example_id in example_ids:  # each example
            contents.append([example_id, num_examples, example_ids - set([example_id]), [example_id]])
#        for animal_id in set(data.Stack().fetch('animal_id')): # each animal
#            val_examples = (data.Stack() & {'animal_id': animal_id}).fetch('example_id')
#            contents.append([animal_id, num_examples, example_ids - set(val_examples), val_examples])
        return contents


@schema
class TrainedModel(dj.Computed):
    definition = """ # trained model and logs
    -> params.ModelParams
    -> params.TrainingParams
    -> Split
    ---
    train_loss:             longblob        # training loss per example
    val_loss:               longblob        # validation loss per epoch
    lr_history:             longblob        # learning rate per epoch
    diverged:               boolean         # whether the loss diverged during training (went to nan or inf)
    best_model:             longblob        # dictionary with trained weights
    best_epoch:             int             # epoch at which best_val_loss was achieved
    best_val_loss:          float           # best validation loss achieved
    best_train_loss:        float           # training loss averaged over all examples in best_epoch
    final_model:            longblob        # model in the final epoch
    final_val_loss:         float           # final loss in the validation set
    final_train_loss:       float           # training loss averaged over all examples in final_epoch
    training_ts=CURRENT_TIMESTAMP:  timestamp
    """
    def make(self, key):
        """ Trains the specified  model using SGD with Nesterovs' Accelerated Gradient.
        Mean cross-entropy loss over all voxels.
        """

        log('Training', key['model_hash'], 'with hyperparams', key['training_hash'],
            'using', key['val_id'], 'for validation.')
        train_params = (params.TrainingParams() & key).fetch1()

        # Set random seeds
        torch.manual_seed(train_params['seed'])
        np.random.seed(train_params['seed'])

        # Get datasets
        train_examples, val_examples = (Split() & key).fetch1('train_examples', 'val_examples')
        train_transform = Compose([transforms.RandomCrop(), transforms.RandomRotate(),
                                   transforms.RandomHorizontalFlip(), transforms.ContrastNorm(),
                                   transforms.Copy()])
        val_transform = Compose([transforms.RandomCrop((225, 512, 512)), transforms.ContrastNorm()])
        dsets = {'train': datasets.SegmentationDataset(train_examples, train_transform),
                 'val': datasets.SegmentationDataset(val_examples, val_transform)}
        dataloaders = {k: DataLoader(dset, shuffle=True, num_workers=4) for k, dset in dsets.items()}

        # Get model
        net = params.ModelParams.build_model(key)
        net.init_parameters()

        # Move network to GPU
        net.cuda()
        net.train()

        # Declare optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=train_params['learning_rate'],
                              momentum=train_params['momentum'], nesterov=True,
                              weight_decay=train_params['weight_decay'])

        # Initialize some logs
        train_loss = []
        val_loss = []
        lr_history = []

        best_model = net
        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(1, train_params['num_epochs'] + 1):
            log('Epoch {}:'.format(epoch))

            # Record learning rate
            lr_history.append(optimizer.param_groups[0]['lr'])

            # Loop over training set
            for volume, label in dataloaders['train']:
                # Move variables to GPU
                volume, label = Variable(volume.cuda()), Variable(label.cuda())

                # Zero the gradients
                net.zero_grad()

                # Compute loss
                output = net(volume) # 1 x num_classes x d x h x w
                vectorized = output.view(output.shape[1], -1).t() # n x num_classes
                loss = criterion(vectorized, label.view(-1))

                # Record training loss
                loss_ = loss.data[0] # float
                log('Training loss:', loss_)
                train_loss.append(loss_)

                # Check for divergence
                if np.isnan(loss_) or np.isinf(loss_):
                    log('Error: Loss diverged!')
                    log('Inserting results...')
                    results = key.copy()
                    results['train_loss'] = train_loss
                    results['val_loss'] = val_loss
                    results['lr_history'] = lr_history
                    results['diverged'] = True # !!
                    results['best_model'] = {k: v.cpu().numpy() for k, v in best_model.state_dict().items()}
                    results['best_epoch'] = best_epoch
                    results['best_val_loss'] = best_val_loss
                    results['best_train_loss'] = compute_loss(best_model, dataloaders['train'], criterion)
                    results['final_model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
                    results['final_val_loss'] = compute_loss(net, dataloaders['val'], criterion)
                    results['final_train_loss'] = compute_loss(net, dataloaders['train'], criterion)
                    self.insert1(results)
                    return -1

                # Backprop
                loss.backward()
                optimizer.step()

            # Record validation loss
            val_loss_ = compute_loss(net, dataloaders['val'], criterion)
            log('Validation loss:', val_loss_)
            val_loss.append(val_loss_)

            # Reduce learning rate
            if ((train_params['lr_schedule'] == 'val_loss' and val_loss_ > best_val_loss)
                or train_params['lr_schedule'] == 'epoch'):
                log('Reducing learning rate...')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= train_params['lr_decay']

            # Save best model
            if val_loss_ < best_val_loss:
                log('Saving best model...')
                best_val_loss = val_loss_
                best_model = net
                best_epoch = epoch

        # Insert results
        log('Inserting results...')
        results = key.copy()
        results['train_loss'] = train_loss
        results['val_loss'] = val_loss
        results['lr_history'] = lr_history
        results['diverged'] = False
        results['best_model'] = {k: v.cpu().numpy() for k, v in best_model.state_dict().items()}
        results['best_epoch'] = best_epoch
        results['best_val_loss'] = best_val_loss
        results['best_train_loss'] = compute_loss(best_model, dataloaders['train'], criterion)
        results['final_model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        results['final_val_loss'] = val_loss_
        results['final_train_loss'] = compute_loss(net, dataloaders['train'], criterion)
        self.insert1(results)

    def load_model(key, best_or_final='best'):
        recarray = (TrainedModel() & key).fetch1('{}_model'.format(best_or_final))
        state_dict = {k: torch.from_numpy(recarray[k][0]) for k in recarray.dtype.names}

        net = params.ModelParams.build_model(key)
        net.load_state_dict(state_dict)

        return net