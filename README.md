# Train a Japanese Languge Model with JavaScript

JavaScriptを使ってローカルで日本語言語モデルを訓練しましょう！

This repo hosts educational scripts for traning a Japanese language model.

This is slightly modified from the [train-model-with-js](https://github.com/frost-beta/train-model-with-js)
repo with a few changes:

* The tokenizer is replaced by [llm-jp-tokenizer](https://github.com/llm-jp/llm-jp-tokenizer)
  (`code10K_en20K_ja30K.ver2.2_hf_fast.b4`), which is specialized for Japanese
  and English.
* The `vocab_size` parameter in `config.json` is changed to match the tokenizer.

## Platform

Only Macs with Apple Silicon are supported.

## Preparations

First clone this repo and install dependencies:

```sh
git clone https://github.com/frost-beta/train-japanese-llama3-js.git
cd train-japanese-llama3-js
npm install
```

Then download the Japanese dataset for training, we are using
[llm-book/aio-passages](https://huggingface.co/datasets/llm-book/aio-passages):

```sh
npm install -g @frost-beta/huggingface
huggingface download --revision refs/convert/parquet datasets/llm-book/aio-passages
```

## Training

To start training, just pass the paths of parquet files to the `train.js`
script:

```sh
node train.js aio-passages/default/train/*
```

For M3 Max with >= 32GB RAM, it takes about 10 hours to train all the data for
one time.

After the training is done, a `weights.safetensors` file will be written, and
you can start generating text with it:

```sh
npm install -g llama3
llama3-generate .
```

To provide the model for inference engine to consume, package following files:

* `config.json`
* `weights.safetensors`
* `tokenizer.json`
* `tokenizer_config.json`

## Example Model

To check what you can get before actual tranining, I have uploaded weights to
[Hugging Face](https://huggingface.co/frost-beta/Llama3-33.5M-Japanese) which
was trained on a Macbook Pro with M3 Max for 10 hours.

```sh
$ npm install -g @frost-beta/huggingface llama3
$ huggingface download frost-beta/Llama3-33.5M-Japanese
$ llama3-generate Llama3-33.5M-Japanese/ F-45は
F-45は日本が開発する軍用機を発売するのが初めてで、1970年代から70年代にかけて広く
用いられた。1979年には『深音振万音時代』の一方で最も人気のあるソ連空軍がアメリカ
陸軍と協議しており、イギリスは軍手に多少の厳しい開国拒否を繰り返したが、特に旧ロ
シア帝国では軍手を押しとどめることはなかった。そのため、N-46は南アフリカの中部、
南大西洋の西岸地域の大陸などにあるとされている。1970年代後半になると、B-46は長距
離の戦車を発見した。
```

## License

Scripts: Public domain

Tokenizer: [Apache License 2.0](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/LICENSE)
