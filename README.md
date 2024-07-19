# Fine Tune Qwen2 into Translator with JavaScript

This repo hosts educational scripts that fine tune the
[Qwen2](https://github.com/QwenLM/Qwen2) model into a English-Chinese translator.

We are choosing this task because:

1. Qwen2 has a small 0.5B pretrained model.
2. Qwen2 is multi-lingual and can handle tranlation tasks.
3. There are high quality English ↔ Chinese corpuses.

## Platform

Only Macs with Apple Silicon are supported.

## How Does it Work

## Preparations

Clone this repo and install dependencies:

```sh
git clone https://github.com/frost-beta/fine-tune-qwen2-js.git
cd fine-tune-qwen2-js
npm install
```

Download the [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) model:

```sh
npm install -g @frost-beta/huggingface
huggingface download Qwen/Qwen2-0.5B
```

Download the [coct-en-zh-tw-translations](https://huggingface.co/datasets/zetavg/coct-en-zh-tw-translations-twp-300k)
dataset for training:

```sh
huggingface download datasets/zetavg/coct-en-zh-tw-translations-twp-300k
```

## Training

Pass the paths of of model and parquet files to the `train.js` script:

```sh
node train.js Qwen2-0.5B coct-en-zh-tw-translations-twp-300k/data/*
```

For M3 Max with >= 64GB RAM, it takes about 10 hours to train all the data for
2 times. For machines with less RAM, you can change the `batchSize` to a smaller
number to reduce RAM usages.

After the training is done, a `fine-tuned.safetensors` file will be written.

## Using the Fine Tuned Model

## License

Scripts: Public domain

Qwen2-0.5B: Apache 2.0
