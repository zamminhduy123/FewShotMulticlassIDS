# FewShot Multiclass IDS in IVN

This code performs a few-shot multi-class classification of the intrusion detection dataset in the in-vehicle network. Using the CarHacking (CHRL) dataset and the ROAD dataset.

## Usage

For the ROAD dataset, process the log file into the CSV using (remember to change the path config)

```
run the notebook ./src/preprocess.ipynb
```

## Preprocess

To process CSV dataset into windows of messages

> Please update feature in `process_data` function in `data_windowing.py` if you want more features

### Process Car Hacking dataset

```
python ./src/data_splitting/data_windowing.py \
    --chd True \
    --is-mar False \
    --process-type <your_process_type_name> \
    --window-size <your_window_size> \
    --window-size <your_step> \
    --feature <number_of_feature> \
    --test-size 0.9 \
    --val-split True \
    --val-size 0.5 \
    --extra 90test_50val \
    --car-hacking-path <your_chd_csv_path> \
    --save-path <your_save_path> \
```

### Process ROAD Fabrication dataset

```
python ./src/data_splitting/data_windowing.py \
    --chd True \
    --is-mar False \
    --process-type <your_process_type_name> \
    --window-size <your_window_size> \
    --window-size <your_step> \
    --feature <number_of_feature> \
    --test-size 0.9 \
    --val-split True \
    --val-size 0.5 \
    --extra 90test_50val \
    --road-data-path <your_road_csv_path> \
    --save-path <your_save_path> \
```

### Process ROAD Masquerade dataset

```
python ./src/data_splitting/data_windowing.py \
    --chd True \
    --is-mar True \
    --process-type <your_process_type_name> \
    --window-size <your_window_size> \
    --window-size <your_step> \
    --feature <number_of_feature> \
    --test-size 0.9 \
    --val-split True \
    --val-size 0.5 \
    --extra 90test_50val \
    --road-data-path <your_road_csv_path> \
    --save-path <your_save_path> \
```

## Training

Please change the path to your dataset in `dataset.py`

**Note**: If train-shot == 0 & test-shot == 0 the code will perform "Episodic training".

```bash
python ./src/main.py \
    --dataset 1 \
    --embedding-dims 64 \
    --d-model 10 \
    --layers 10 \
    --num-class 12 \
    --norm True\
    --batch-size 128 \
    --epochs 300 \
    --optimizer AdamW \
    --scheduler-warmup 100 \
    --lr 0.0001 \
    --window-size 16 \
    --step-size 1 \
    --features 11 \
    --split <your_split_name> \
    --train-shot 0 \
    --test-shot 0 \
    --circle-loss \
    --ood-loss \
    --checkpoint-dir your_path \
    --pretrained-path your_path \
```
