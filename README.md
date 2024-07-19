# Fine-Tune Qwen2 into a Translator with JavaScript

This repo hosts educational scripts that fine-tune the
[Qwen2](https://github.com/QwenLM/Qwen2) model into a English-Chinese translator.

We are choosing this task because:

1. Qwen2 has a small 0.5B pretrained model.
2. Qwen2 is multi-lingual and can handle tranlation tasks.
3. There are high quality English ↔ Chinese corpuses.

## Platform

Only Macs with Apple Silicon are supported.

## How Does it Work

The corpus has 2 columns for English and Chinese text:

| en                                                                                  | ch                                                                   |
|-------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| After being treated with carbon dioxide, the cabbage lettuce tripled in production. | 其中以結球萵苣的效果最為顯著，通過適量二氧化碳後的產量為以前的三倍。 |

In order to train a language model with the data we are concatenating the 2
columns into one text:

```
<|im_start|>
Translate to Chinese:
After being treated with carbon dioxide, the cabbage lettuce tripled in production.
<|im_end|>
<|im_start|>
其中以結球萵苣的效果最為顯著，通過適量二氧化碳後的產量為以前的三倍。
<|im_end|>
```

The `<|im_start|>` and `<|im_end|>` are special tokens used for training chat
bots, and we are using them to separate the Chinese and English text.

After the training is done, we can use a prompt like below to feed English to
the fine-tuned model:

```
<|im_start|>
Translate to Chinese:
Yehliu is peculiar not only in its beautiful rocks, but also in its tropical fish, corals and shells.
<|im_end|>
<|im_start|>
```

and get Chinese output as text generation:

```
野柳濱海也有奇妙的生物，像木瓜魚、紅龜鱘、海星、珊瑚和貝殼。
<|im_end|>
```

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

## Using the Fine-Tuned Model

Overwrite the Qwen2's weights with our fine-tuned weights:

```sh
mv fine-tuned.safetensors Qwen2-0.5B/model.safetensors
```

Feed the model with some English sentences and get Chinese output:

```sh
echo 'Anxiety is the dizziness of freedom.' | node translator.js Qwen2-0.5B
```

## License

Scripts: Public domain

Qwen2-0.5B: Apache 2.0
