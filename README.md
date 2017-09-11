# bl3d
Cell segmentation in 3D GCaMP structural recordings

# Localization
We detect cells using a 3-D ConvNet trained to predict cell locations. Labels are generated using the red channel of mCherry expressing mice ('watermelon' mice). The network output is a 3-d probability map with a score per voxel.

# Centroid estimates
We threshold the probability heatmap produced by the network at p > 0.5, extract connected components and compute the center of mass for each component.

# Segmentation
We produce the final segmentation by computing the correlation of each centroid with voxels in a 20 x 20 x 20 window and threshold at sigma > 0.5.
