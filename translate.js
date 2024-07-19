#!/usr/bin/env node

import {readFileSync} from 'node:fs'
import {core as mx} from '@frost-beta/mlx'
import {TokenizerLoader} from '@lenml/tokenizers'

import Model from './model.js'

if (process.argv.length < 3) {
  console.error('Usage: train.js /model-dir')
  process.exit(0)
}

const modelDir = process.argv[2]

// Load model.
const config = JSON.parse(readFileSync(`${modelDir}/config.json`))
const model = new Model(config)
model.loadWeights(`${modelDir}/model.safetensors`)

// Load tokenizer.
const tokenizer = TokenizerLoader.fromPreTrained({
  tokenizerJSON: JSON.parse(readFileSync(`${modelDir}/tokenizer.json`)),
  tokenizerConfig: JSON.parse(readFileSync(`${modelDir}/tokenizer_config.json`)),
})
const imStart = '<|im_start|>'
const imEnd = '<|im_end|>'
const eos = '<|endoftext|>'
const [imStartToken, imEndToken, eosToken] = tokenizer.encode(imStart + imEnd + eos)

// Read data from stdin until EOF.
let input = ''
process.stdin.on('data', (chunk) => input += chunk)
process.stdin.on('end', main)

function main() {
  // Construct prompt.
  const prompt = `${imStart}Translate to English:\n${input}${imEnd}${imStart}`
  const promptTokens = tokenizer.encode(prompt)

  // Generate output.
  const temperature = 0.9
  for (const token of step(promptTokens, model, temperature)) {
    const char = tokenizer.decode([token])
    process.stdout.write(char)
  }
  process.stdout.write('\n')
}

// Generator for text generation.
function* step(promptTokens, model, temperature) {
  const forward = (y) => {
    let logits = model.forward(mx.array([y], mx.int32))
    logits = logits.index(mx.Slice(), -1, mx.Slice())
    const token = sample(logits, temperature)
    return token.item()
  }

  let tokens = promptTokens
  while (true) {
    const token = mx.tidy(() => forward(tokens))
    if (token == imEndToken || token == eosToken || tokens.length > 256)
      break
    yield token
    tokens.push(token)
  }
}

// Pick the best token from logits.
function sample(logits, temperature) {
  return mx.random.categorical(mx.multiply(logits, 1 / temperature))
}
