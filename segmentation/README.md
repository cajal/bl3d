# bl3d
Cell segmentation in 3D GCaMP structural recordings.

We segment cells using a 3-d fully convolutional network. Labels are generated using the red channel of GCaMP6 + mCherry expressing mice ('watermelon' mice). The network outputs two voxel-wise probability maps: one represents the estimated centroids of each cell and the other segments the cells; both are binary segmentations. To produce instance segmentations, we threshold the segmentation map and apply compact watershed using peaks in the centroid map like markers. See the [demo](Demo.ipynb) for results. See the references for a similar approach applied to embryonic cells.

### References
[1]  Convolutional Neural Network-Based Instance Segmentation Algorithm to Acquire Quantitative Criteria of Early Mouse Development
Yuta Tokuoka, Takahiro G Yamada, Noriko Hiroi, Tetsuya J Kobayashi, Kazuo Yamagata, Akira Funahashi
bioRxiv 324186; doi: https://doi.org/10.1101/324186 