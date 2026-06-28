import { AutoProcessor, Qwen3_5ForConditionalGeneration, env, Tensor } from "@huggingface/transformers";
import { createInterface } from "node:readline";
import { resolve } from "node:path";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.proxy = false;

const args = parseArgs(process.argv.slice(2));
const modelDir = resolve(args["model-dir"] ?? "runtime/models/Qwen3___5-0___8B-ONNX");

let processorPromise;
let modelPromise;

async function loadProcessor() {
  processorPromise ??= AutoProcessor.from_pretrained(modelDir, { local_files_only: true });
  return processorPromise;
}

async function loadModel() {
  modelPromise ??= Qwen3_5ForConditionalGeneration.from_pretrained(modelDir, {
    local_files_only: true,
    dtype: {
      embed_tokens: "q4",
      decoder_model_merged: "q4",
      vision_encoder: "q4",
    },
    device: "cpu",
  });
  return modelPromise;
}

async function generateText({ messages, temperature = 0.1, max_tokens = 768 }) {
  const processor = await loadProcessor();
  const model = await loadModel();
  const text = processor.apply_chat_template(normalizeMessages(messages), {
    add_generation_prompt: true,
    enable_thinking: false,
  });
  const inputs = await processor(text);
  const inputLength = inputs.input_ids.dims.at(-1);
  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: Math.max(1, Math.min(Number(max_tokens) || 768, 768)),
    do_sample: Number(temperature) > 0,
    temperature: Number(temperature) || 0.1,
    top_k: 20,
    top_p: 0.95,
  });
  const generated = sliceGeneratedTokens(outputs, inputLength);
  const decoded = processor.batch_decode(generated, {
    skip_special_tokens: true,
  });
  return String(decoded?.[0] ?? "").replace(/^\s*<think>\s*<\/think>\s*/s, "").trim();
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    return [{ role: "user", content: "" }];
  }
  return messages.map((message) => ({
    role: message.role === "system" || message.role === "assistant" || message.role === "user" ? message.role : "user",
    content: typeof message.content === "string" ? message.content : String(message.content ?? ""),
  }));
}

function sliceGeneratedTokens(outputs, inputLength) {
  if (outputs && typeof outputs.slice === "function") {
    return outputs.slice(null, [inputLength, null]);
  }
  if (outputs instanceof Tensor && Array.isArray(outputs.dims) && outputs.dims.length === 2) {
    const [batch, totalLength] = outputs.dims;
    const width = Math.max(0, totalLength - inputLength);
    const data = outputs.data.slice(inputLength, totalLength);
    return new Tensor(outputs.type, data, [batch, width]);
  }
  return outputs;
}

async function handleRequest(request) {
  try {
    const text = await generateText(request);
    writeJson({ id: request.id, ok: true, text });
  } catch (error) {
    console.error(error?.stack || error?.message || String(error));
    writeJson({ id: request.id, ok: false, error: error?.message || String(error) });
  }
}

function writeJson(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0;index < argv.length;index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) {
      continue;
    }
    const key = item.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

if (args["self-test"]) {
  const text = await generateText({
    id: "self-test",
    messages: [{ role: "user", content: "请用一句话介绍你自己。" }],
    temperature: 0.1,
    max_tokens: Number(args["max-new-tokens"] ?? 16),
  });
  writeJson({ id: "self-test", ok: true, text });
  process.exit(0);
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  const trimmed = line.trim();
  if (!trimmed) {
    continue;
  }
  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (error) {
    writeJson({ id: null, ok: false, error: `invalid json: ${error?.message || error}` });
    continue;
  }
  await handleRequest(request);
}
