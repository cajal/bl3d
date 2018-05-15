# Models
Linear filter or dictionary of filters are hard to learn with SGD.
Fully-convolutional network learn easily. Batchnorm makes trainig more stable, final results better (0.01-0.07 IOU differential).

# Hyperparameter search
Using the same learning rate across entire training rather than reducing learning rate every epoch that the loss increases (0.04-0.06 IOU differential).
    Stepwise decreases at epoch 100 and 200, for instance may even be better.
Weighting the loss to increase the value of positive classes makes it better (0.5-0.1 without batchnorm, 0.01-0.02 when using batchnorm). As batchnorm, this may be easing training.
Low close-to-zero lambdas have better results. lambdas >=0.1 cause divergence.
High learning rates seem better 0.01 to 0.1 even though 0.1 starts to make noisier loss curves. Didn't try anything hiogher than 0.1. Maybe 0.1 needs more time to converge.

# Caveat
Same example used for early stopping as for hyperparameter search (but not in training).

# Conclusions
Simpler versions get to 0.35 IOU. Either batchnorm or weighted loss will increase IOu for ~0.35 to 0.45. If using both increase it to 0.46.
