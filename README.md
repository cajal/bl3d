# bl3d
Cell detection and segmentation in 3D GCaMP structural recordings

# Structural Instance Segmentation (3-d)
We segment cells using a 3-d Mask R-CNN (He et al, 2018). Labels are generated using the red channel of GCaMP6 + mCherry expressing mice ('watermelon' mice). For each predicted mask, the network outputs a cellness likelihood and a probability map with a score per voxel; we threshold these to produce binary segmentations.

# Functional Segmentation (2-d)
To obtain masks for functional planes, we register the planes to the structural scan, correlate the pixels in a 20 x 20 window around the centroid of each intersecting mask and threshold at sigma > 0.5 to produce the final segmentation.
