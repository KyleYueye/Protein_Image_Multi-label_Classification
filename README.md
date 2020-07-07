# Protein Image Multi-label Classification
## Project code for CS385 Machine Learning in SJTU

### Project Requirements
- python >= 3.6
- pytorch = 1.5.1
- tensorflow = 2.0.0
- opencv-python = 4.2.0.34

### Project structure

* `main.py`: main training file, please run `python main.py`. And notice the file paths in `main.py`. 
* `predict.py`: a predict class and 3 member functions
    1. `random_predict()`, which can pick an image randomly from the test dataset and predict it. The star sign(*) means the groundtruth class.
    2. `specific_predict(img)`, which reads an image and gives the result of the prediction.
    3. `overall_predict()`, which can predict all of the images in the test set and give the `roc_auc`, `macro_f1`, `micro_f1` score of the prediction.
* `manage_data.py`: file of spliting datasets.
*  `history.py`: file of drawing history graph by using TensorBoard or matplotlib.
* `cmd`: file of command line instructions.
* `logs/`: log files of training.
* `models/`: trained models.

### Results
We split the dataset by `train:test:valid = 8:1:1`. And the current result comparing to the baseline offered by TA:


|               | macro_f1  | micro_f1  |  roc_auc  |
|  ------       |   ------  |  ------   |  ------   |
| baseline      |  57.18%   | 73.37%    |  93.82%   |
| my            |  69.23%   | 72.21%    |  94.23%   |
| **final result**  |  **74.22%**   | **79.89%**    |  **96.19%**   |


