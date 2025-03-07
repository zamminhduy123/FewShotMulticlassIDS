# FewShot Multiclass IDS in IVN

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
python ./src/main.py \
    --dataset 1 \
    --embedding-dims 64 \
    --d-model 10 \
    --layers 10 \
    --num-class 12 \
    --norm \
    --batch-size 128 \
    --epochs 600 \
    --optimizer AdamW \
    --scheduler CosineWarmup100 \
    --lr 0.0001 \
    --window-size 16 \
    --step-size 1 \
    --features 11 \
    --split ss_11_2d_no \
    --train-shot 5 \
    --test-shot 0 \
    --circle-loss \
    --ood-loss \
    --checkpoint-dir your_path \
    --pretrained-path your_path \
```
