# FewShotMulticlassIDS

This code performs a few-shot multi-class classification of the intrusion detection dataset in the in-vehicle network. Using the CarHacking (CHRL) dataset and the ROAD dataset.

## Usage

For the ROAD dataset, process the log file into the CSV using (remember to change the path config)
```
run the notebook ./src/preprocess.ipynb
```

To process CSV dataset into windows of messages (Adjust config at the end of the file)
```
python ./src/data_splitting/data_windowing.py
```

To run the training 

```bash
python ./src/main.py
```
