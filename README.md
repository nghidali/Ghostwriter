# PPLM

This repository contains code to run the Plug and Play Language Model (PPLM), as described in this **[blog post](https://eng.uber.com/pplm)** and **[arXiv paper](https://arxiv.org/abs/1912.02164)**. A **[demo](https://transformer.huggingface.co/model/pplm)** and **[Colab notebook](https://colab.research.google.com/drive/1Ux0Z4-ruiVtJ6jUk98uk6FqfvGHCOYL3)** are also available.

PPLM is also integrated into the **[🤗/Transformers](https://github.com/huggingface/transformers/tree/master/examples/pplm)** repository.

![header image](./imgs/headfigure.png)

## Plug and Play Language Models: a Simple Approach to Controlled Text Generation
Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)

PPLM allows a user to flexibly plug in one or more tiny attribute models representing the desired steering objective into a large, unconditional language model (LM). The method has the key property that it uses the LM _as is_—no training or fine-tuning is required—which enables researchers to leverage best-in-class LMs even if they do not have the extensive hardware required to train them.

See also our [arXiv paper](https://arxiv.org/abs/1912.02164), [blog post](https://eng.uber.com/pplm), and try it out for yourself with no setup using the [Colab notebook](https://colab.research.google.com/drive/1Ux0Z4-ruiVtJ6jUk98uk6FqfvGHCOYL3).

## Setup

```bash
pip install -r requirements.txt
```

## Citation
```
@inproceedings{
Dathathri2020Plug,
title={Plug and Play Language Models: A Simple Approach to Controlled Text Generation},
author={Sumanth Dathathri and Andrea Madotto and Janice Lan and Jane Hung and Eric Frank and Piero Molino and Jason Yosinski and Rosanne Liu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1edEyBKDS}
}
```

## PPLM-BoW

### Example command for bag-of-words control

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample
```

### Tuning hyperparameters for bag-of-words control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model.

2. If the language being generated is repetitive (For e.g. "science science experiment experiment"), there are several options to consider: </br>
	a) Reduce the `--stepsize` </br>
	b) Increase `--kl_scale` (the KL-loss coefficient) or decrease `--gm_scale` (the gm-scaling term) </br>
	c) Add `--grad-length xx` where xx is an (integer <= length, e.g. `--grad-length 30`).</br>


## PPLM-Discrim

### Example command for discriminator based sentiment control

```bash
python run_pplm.py -D sentiment --class_label 2 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

### Tuning hyperparameters for discriminator control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model.

2. Use `--class_label 3` for negative, and `--class_label 2` for positive


The discriminator and the GPT-2 model in the root directory are different from those used for the analysis in the paper. Code and models corresponding to the paper can be found [here](https://github.com/uber-research/PPLM/tree/master/paper_code).

### Making a custom discriminator

1. Create a tab separated text file with the emotional label and the text on each line ex: "3\thello I'm Natalie\n4\tI'm so happy\n"
2. Rename this as a tsv, for example natOutput.tsv
3. run the following to generate a bunch of files, the ones we use are "generic_classifier_head_epoch_10.pt" and "generic_classifier_head_meta.json"
```bash
python3 run_pplm_discrim_train.py --dataset generic --dataset_fp natOutput.tsv --cached --save_model
```
4. Now we will plug this into our pplm model as so
```bash
python3 run_pplm.py -D generic --discrim_weights "generic_classifier_head_epoch_10.pt" --discrim_meta "generic_classifier_head_meta.json" --class_label 4 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

### Running Nat's DailyDialog discriminator
```bash
python3 run_pplm.py -D generic --discrim_weights "emotion.pt" --discrim_meta "emotion.json" --class_label 4 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```
The class label is  { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}

### Choosing the Best Sample with driver.py

When running PPLM, the user decides how many samples to output. Not every sample is semantically, gramatically, or emotionally accurate.
Running `driver.py` will output the user with the most semantically, gramatically, and emotionally correct sample from the n outputs.

To run this driver, just run ```python driver.py --emotion [chosen-emotion] --input [cond-text]```

driver.py first combs through all the samples and removes extraneous words from ends of the samples.
Then grammarbot, WordNet, and IBM Watson's Tone Analyzer are applied to all the samples.
We get scores from all 3 systems and normalize the weights before applying weights.

The default weights are:
1. Sentiment Analysis = 1
2. Semantic Similarity = 0.8
3. Grammar = 0.3

Grammar is weighted much lower that the other weights because the sentences are corrected for grammar and are returned. Therefore, we are focused more on how emotional and related the samples are to the conditional text input.

A higher weight means the scale is more important. A higher score from all 3 metrics means the sample is more accurate.

## Tuning parameters for driver.py

1. `--emotion` tells the driver which emotion to look for in the samples (choose between: )
2. `--input` is the conditional text the user inputs when running run_pplm.py
3. `--output_file` is the file where the outputs from PPLM are stored (run_pplm.py by default, stores the samples at samples.txt)
4. `--sa_weight` is the weight of sentiment analysis. Increasing this will weight IBM Watson's tone analyzer more.
5. `--semantic_weight` is the weight of WordNet. Increasing this will weight semantic similarity more.
6. `--grammar_weight` is the weight of Grammarbot. Increasing this will weight grammar accruacy more.

