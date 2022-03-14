# HateSpeechDetection

## Steps to generate SOC Hierarchical Explanation scores

```
docker pull aggarwalpiush/hierarchical_explanation:v1
```

```
docker run -it --gpus all --name {CONTAINER_NAME} -v {LOCATION_SCORE_FILE}:/home/piush/hierarchical-explanation-neural-sequence-models/hiexpl/outputs/davidson/soc_results aggarwalpiush/hierarchical_explanation:v1 /bin/bash
```

```
conda activate hiexpl-env 
```

```
cd /home/piush/hierarchical-explanation-neural-sequence-models/hiexpl
```

Move your train, dev and test files to container at path /home/piush/hierarchical-explanation-neural-sequence-models/hiexpl/.data/davidson/tsv/

The Format of the files should be like 

```
sentence\tlab\n
```

Step 1:  train the LSTM model on your dataset and generate vocabulary pickle file

```
export model_path=models/davidson_lstm
rm -rf vocab/*
rm -rf models/*
rm -rf outputs/davidson/soc_results/*
python train.py --task davidson --save_path models/${model_path} --no_subtrees --lr 0.0005
```

Step 2: Finetune the language model on the provided corpus

```
export lm_path=models/davidson_lstm_lm
python -m lm.lm_train --task davidson --save_path models/${lm_path} --no_subtrees --lr 0.0002

```

Step 3: Generate the explanation score file. 

Note: It is time consuming and depends for many sentences you want to generate explanation scores. For first 100 training samples following command can be used. 

```
export algo=soc # or scd
export exp_name=.davidson_demo
python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 100 --exp_name ${exp_name} --task davidson --explain_model lstm --nb_range 10 --sample_n 20 --dataset train
```

For whole text file remove the stop tag

```
python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --exp_name ${exp_name} --task davidson --explain_model lstm --nb_range 10 --sample_n 20 --dataset train
```

Raise issues in case of any clarification or issue raised.

