# bl3d
Cell segmentation in 3D GCaMP structural recordings.

# Structural Segmentation (3-d)
We segment cells using a 3-d ConvNet. Labels are generated using the red channel of GCaMP6 + mCherry expressing mice ('watermelon' mice). The network outputs a probability map with a score per voxel; we threshold it to produce a binary segmentation. We use watershed segmentation to produce instance segmentations.

# Functional Segmentation (2-d)
To obtain masks for functional planes, we register the planes to the structural scan, correlate the pixels in a 20 x 20 window around the centroid of each intersecting mask and threshold at sigma > 0.5 to produce the final segmentation.
