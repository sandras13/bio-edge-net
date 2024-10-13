# BioEdgeNet - A Lightweight Deep Neural Network for Stress Recognition on Edge Devices

Results: https://docs.google.com/spreadsheets/d/1MerMBSwIddq80V7AEy5kXrv8e_uc7PLUy5fTDZmDKW8/edit?usp=sharing

To execute the code, the datasets must be saved in .csv format. We concentrated only on the PPG and ACC data, extracting only those parameters from the original dataset and compiling them into joined .csv files. These files also include information about the subjects and their stress levels to facilitate proper splitting into sliding windows.

Initially, the dataset, the selected signals, and the quantization methods are determined using specified indices. Training parameters are stored in a variable named params.

The signals are divided into sliding windows, ensuring that each window contains the most frequently used label from the dataset, while also preventing signals from different subjects from being combined in the same window. This approach is based on the understanding that stress levels may fluctuate during measurements, but the subject will remain constant.

The model is then constructed and trained, with options for pruning included in the code, although they were not utilized in the final version. The accuracy, F1 score, precision, and recall are computed by averaging the results from 5-fold cross-validation.
