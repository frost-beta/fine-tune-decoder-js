# Fine-tune a decoder-only model with JavaScript

This repo hosts educational scripts that fine-tune the decoder-only model
[Qwen2](https://github.com/QwenLM/Qwen2) into a Chinese-English translator.

Qwen2 is multi-lingual and has a small 0.5B pretrained version, which makes it
very useful to be fine-tuned for specialized tasks. We are traning it into a
simple translator, and you can compare this repo with :construction: to see
the differences between traning decoder-only and encoder-only models.

## Platform

Only Macs with Apple Silicon are supported.

## How does it work

The corpus has 2 columns for English and Chinese text:

| en                                                                                  | ch                                                                   |
|-------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| After being treated with carbon dioxide, the cabbage lettuce tripled in production. | 其中以結球萵苣的效果最為顯著，通過適量二氧化碳後的產量為以前的三倍。 |

In order to train a language model with the data we are concatenating the 2
columns into one text:

```
<|im_start|>
Translate to English:
其中以結球萵苣的效果最為顯著，通過適量二氧化碳後的產量為以前的三倍。
<|im_end|>
<|im_start|>
After being treated with carbon dioxide, the cabbage lettuce tripled in production.
<|im_end|>
```

The `<|im_start|>` and `<|im_end|>` are special tokens used for training chat
bots, and we are using them to separate the Chinese and English text.

After the training is done, we can use a prompt like below to feed Chinese to
the fine-tuned model:

```
<|im_start|>
Translate to English:
野柳濱海也有奇妙的生物，像木瓜魚、紅龜鱘、海星、珊瑚和貝殼。
<|im_end|>
<|im_start|>
```

and get English output through text generation:

```
Yehliu is peculiar not only in its beautiful rocks, but also in its tropical fish, corals and shells.
<|im_end|>
```

## Preparations

Clone this repo and install dependencies:

```sh
git clone https://github.com/frost-beta/fine-tune-decoder-js.git
cd fine-tune-decoder-js
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

## Use the fine-tuned model

Overwrite the Qwen2's weights with our fine-tuned weights:

```sh
mv fine-tuned.safetensors Qwen2-0.5B/model.safetensors
```

Feed the model with some Chinese sentences and get English output:

```sh
echo '微軟Windows作業系統19日大當機，災情遍及全球' | node translator.js Qwen2-0.5B
```

## What's next

After getting familiar with traning the decoder-only model, you can visit
the :construction: to see how to train a encoder-only model to do the same task.

## License

Scripts: Public domain

Qwen2-0.5B: Apache 2.0
