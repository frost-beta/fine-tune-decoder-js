#!/usr/bin/env node

import fs from 'node:fs/promises'
import path from 'node:path'
import prettyMilliseconds from 'pretty-ms'
import {ParquetGroupReader} from '@frost-beta/parquet-reader'
import {TokenizerLoader} from '@lenml/tokenizers'
import {core as mx, optimizers as optim, nn, utils} from '@frost-beta/mlx'

import Model from './model.js'

if (process.argv.length < 4) {
  console.error('Usage: train.js /model-dir /path/to/train-*.parquet')
  process.exit(0)
}

// Hyperparameters.
const contextSize = 128 + 64

// Traning configs.
const batchSize = 8
const learningRate = 2e-5
const maxRows = Infinity

main(process.argv[2], process.argv.slice(3))

async function main(modelDir, files) {
  // Load model.
  const config = JSON.parse(await fs.readFile(`${modelDir}/config.json`))
  const model = new Model(config)
  model.loadWeights(`${modelDir}/model.safetensors`)

  // Load model's tokenizer.
  const tokenizer = TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(await fs.readFile(`${modelDir}/tokenizer.json`)),
    tokenizerConfig: JSON.parse(await fs.readFile(`${modelDir}/tokenizer_config.json`)),
  })

  // Calculate how many parameters the model has.
  let nparams = 0
  for (const [k, x] of utils.treeFlatten(model.parameters())) {
    nparams += x.size
  }
  console.log(`Fine tuning ${config.model_type} with ${(nparams / 1024 ** 2).toFixed(1)}M parameters.`)

  // Prepare dataset.
  const reader = new ParquetGroupReader(files)
  const totalRows = Math.min(maxRows, await reader.getRowsCount())
  const reportPerIter = Math.max(Math.floor(8 / batchSize * 10), 1)
  console.log('Total rows of data to train:', totalRows)

  // Preprare utils for doing gradient descent.
  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction)
  const optimizer = new optim.AdamW(learningRate)

  // Read batches from the datasets.
  let iter = 0
  let start = Date.now()
  let losses = []
  let lastRow = 0
  for await (const [row, x, y] of await iterateBatches(reader, tokenizer, contextSize, batchSize)) {
    // Use mx.tidy to free all the intermediate tensors immediately.
    mx.tidy(() => {
      // Compute loss and gradients, then update the model.
      const [loss, grads] = lossAndGradFunction(model, mx.array(x, mx.int32), mx.array(y, mx.int32))
      optimizer.update(model, grads)
      mx.eval(model.state, optimizer.state)
      losses.push(loss.item())
      // Keep the states of model and optimizer from getting freed.
      return [model.state, optimizer.state]
    })
    // Report updates.
    if (++iter % reportPerIter === 0) {
      const stop = Date.now()
      const trainLoss = mean(losses)
      const eta = (totalRows - row) / (row - lastRow) * (stop - start)
      console.log(`Iter ${iter}`,
                  `(${(100 * row / totalRows).toFixed(1)}%):`,
                  `Train loss ${trainLoss.toFixed(2)},`,
                  `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(2)},`,
                  `ETA ${prettyMilliseconds(eta, {compact: true})}.`)
      start = Date.now()
      losses = []
      lastRow = row
    }
    if (lastRow > maxRows)
      break
  }
  await reader.close()

  // Save weights on exit.
  const weightsFile = 'fine-tuned.safetensors'
  console.log(`Save weights to ${weightsFile}.`)
  model.saveWeights(weightsFile)
}

// Read datasets from |files|, and generate batches of [inputs, targets].
async function* iterateBatches(reader, tokenizer, contextSize, batchSize) {
  const eosTokenId = tokenizer.encode(tokenizer.getToken('eos_token'))[0]
  let row = 0
  const xBatches = []
  const yBatches = []
  for await (const data of await iterateDataset(reader)) {
    ++row
    // Convert text to tokens.
    const tokens = tokenizer.encode(data)
    // Ignore long tokens, since context is vital for the training, we should
    // not truncate the input.
    const length = tokens.length - 1
    if (length > contextSize)
      continue
    // Fill rest of the input with EOS.
    const paddings = new Array(contextSize - length).fill(eosTokenId)
    xBatches.push(tokens.slice(0, length).concat(paddings))
    yBatches.push(tokens.slice(1, length + 1).concat(paddings))
    // Yield batches with each batch of |batchSize|.
    while (xBatches.length >= batchSize) {
      yield [row, xBatches.splice(0, batchSize), yBatches.splice(0, batchSize)]
    }
  }
}

// The dataset has 2 columns for english and chinese passages, since we are
// training a LLM, combine them into one text and feed to trainer.
async function* iterateDataset(reader, tokenizer) {
  const start = '<|im_start|>'
  const end = '<|im_end|>'
  for await (const data of await reader.getIterator({shuffle: true, chunkSize: 1024})) {
    const [en, zh] = data
    yield `${start}Translate to Chinese:\n${en}${end}${start}${zh}${end}`
  }
}

// Calculate the loss by 1) running the model with the inputs, and 2) then using
// cross entropy function to get the loss between the results and targets.
function lossFunction(model, x, y) {
  const logits = model.forward(x)
  const losses = nn.losses.crossEntropy(logits, y)
  return mx.mean(losses)
}

// Compute the mean value of a JS array.
function mean(array) {
  if (array.length == 0)
    return 0
  return array.reduce((a, b) => a + b) / array.length
}
