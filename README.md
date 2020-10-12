# Data
The ```./evidence_integration``` directory stores the Evidence Integration dataset. Evidence Integration is generated by re-purposing an existing dataset, Evidence Inference v1.0 ([Lehman et al, 2019](http://evidence-inference.ebm-nlp.com/)). Materials from Evidence Inference are stored in ```./evidence_ingreation/materials.zip```. To generate the dataset, Run:
```bash
cd evidence_integration
unzip materials.zip
python generate_evidence_integration.py 
```
This will generate the standard Evdience Integration dataset splits (```train.json```, ```validation.json``` and ```test.json```).

To run our model scripts, further process the dataset splits by running:
```bash
python index_dataset.py
```
This will generate ```indexed_[split]_picos.json``` and ```indexed_[split]_ctxs.json``` for each split.

# Pre-training Data
First, download and ungunzip all the PubMed baseline splits [here](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/). Run:
```bash
cd pretraining_dataset
python preprocess_pubmed_splits.py
```
to parse the xml files and generate the json files for each split.

Then run:
``` bash
python tag_dataset.py # Stanford POS tagger is required to run this script. This will generate collected implicit evidence and contexts in the ./evidence repo.
python process_tages.py # This will process and aggregate all collected evidence
python aggregate_ctxs.py # This will process and aggregate all collected contexts
```
These will generate the final pretraining evidence data ```evidence.json``` and the corresponding contexts ```pmid2ctx.json```.

To run the pretraining scripts, further process the evidence by running:
```bash
python index_dataset.py
```
to generate ```indexed_evidence.json``` and ```indexed_contexts.json``` for pre-training.

# Configuration
Experiments are conducted using Python 3.7.6. 
The computing enviroment is shown in ```requirements.txt```.  

# Usage
The codes are modified from Huggingfaces' Transformers package.

Run ```run_ebmnet.py```:
```bash
$python run_ebmnet.py  -h
usage: run_ebmnet.py [-h] --model_name_or_path MODEL_NAME_OR_PATH --output_dir
                     OUTPUT_DIR [--train_ctx TRAIN_CTX]
                     [--predict_ctx PREDICT_CTX] [--repr_ctx REPR_CTX]
                     [--train_pico TRAIN_PICO] [--predict_pico PREDICT_PICO]
                     [--repr_pico REPR_PICO] [--permutation PERMUTATION]
                     [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
                     [--max_passage_length MAX_PASSAGE_LENGTH]
                     [--max_pico_length MAX_PICO_LENGTH] [--do_train]
                     [--do_eval] [--do_repr] [--evaluate_during_training]
                     [--do_lower_case]
                     [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                     [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                     [--learning_rate LEARNING_RATE]
                     [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                     [--weight_decay WEIGHT_DECAY]
                     [--adam_epsilon ADAM_EPSILON]
                     [--max_grad_norm MAX_GRAD_NORM]
                     [--num_train_epochs NUM_TRAIN_EPOCHS]
                     [--max_steps MAX_STEPS] [--warmup_steps WARMUP_STEPS]
                     [--logging_steps LOGGING_STEPS] [--save_steps SAVE_STEPS]
                     [--eval_all_checkpoints] [--no_cuda] [--overwrite_cache]
                     [--seed SEED] [--local_rank LOCAL_RANK] [--pretraining]
                     [--num_labels NUM_LABELS] [--adversarial]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The path of the pre-trained model.
  --output_dir OUTPUT_DIR
                        The output directory where the model checkpoints and
                        predictions will be written.
  --train_ctx TRAIN_CTX
                        json file for training
  --predict_ctx PREDICT_CTX
                        json for predictions
  --repr_ctx REPR_CTX   json for representatins
  --train_pico TRAIN_PICO
                        json for training
  --predict_pico PREDICT_PICO
                        json for predictions
  --repr_pico REPR_PICO
                        json for representatins
  --permutation PERMUTATION
                        The sequence of intervention, comparison and outcome
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name
  --cache_dir CACHE_DIR
                        Where do you want to store the pre-trained models
                        downloaded from s3
  --max_passage_length MAX_PASSAGE_LENGTH
                        max length of passage.
  --max_pico_length MAX_PICO_LENGTH
                        max length of pico.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_repr             Whether to get representations
  --evaluate_during_training
                        Rul evaluation during training at each logging step.
  --do_lower_case       Set this flag if you are using an uncased model.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --eval_all_checkpoints
                        Evaluate all checkpoints starting with the same prefix
                        as model_name ending and ending with step number
  --no_cuda             Whether not to use CUDA when available
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --seed SEED           random seed for initialization
  --local_rank LOCAL_RANK
                        local_rank for distributed training on gpus
  --pretraining         Whether to do pre-training
  --num_labels NUM_LABELS
                        Number of labels at the last layer. Use 34 in pre-
                        training and 3 in fine-tuning.
  --adversarial         Whether using the adversarial setting.
```

Specifically, run the following codes for pre-training:
```bash
python -u run_eubmnet.py --model_name_or_path ${BIOBERT_PATH} \
--do_train --train_pico pretraining_dataset/indexed_evidence.json --train_ctx pretraining_dataset/index_contexts.json \
--num_labels 34 --output_dir ${PRETRAINED_MODEL} --pretraining --adversarial
```
	
Run the following codes for fine-tuning:
```bash
python -u run_ebmnet.py --model_name_or_path ${PRETRAINED_MODEL} \
--do_train --train_pico evidence_integration/indexed_train_picos.json --train_ctx evidence_integration/indexed_train_ctxs.json \
--do_eval --predict_pico evidence_integration/indexed_validation_picos.json --predict_ctx evidence_integration/indexed_validation_ctxs.json \
--output_dir ${OUTPUT_DIR} 
``` 
