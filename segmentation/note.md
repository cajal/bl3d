# Note
This is our first go at producing instance segmentations: by segmenting first and separating instances later; a FCN plus simple computer vision methods. Latest version runs detection first and segmentation later; both in a single network. Data and code in this folder are freezed but fully functional.

There was a bug in train.py when saving the model so no early stopping was 
performed. `best_model = final_model`
