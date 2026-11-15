// Cross-wire translation for the aggregator: every client wire (OpenAI, Anthropic) to every
// provider wire (OpenAI, Anthropic, Gemini) and back.
//
// The providers echo what they parsed (system text, user text, tool names, max_tokens, stop
// sequences, image, JSON mode), and the env's client compares that echo with what it sent. So
// every translation below is built to preserve the echo exactly:
//   - system text: joined with "\n" exactly as the receiving wire's parser joins it
//   - user text: the last user message's text blocks, joined with "\n"
//   - tool names: in order, from the tool definitions
//   - max_tokens / stop sequences: passed through, never invented
//   - images: the same bytes (base64) or the same URL, so the sha8 matches
//   - JSON mode: only carried on wires that can express it (OpenAI response_format, Gemini
//     responseMimeType); an Anthropic-wire provider echoes json_mode false, so a JSON-mode
//     client request is never routed to one (main.ts filters those candidates out).
//
// Responses are parsed, never proxied: a provider's answer text is carried whole, its usage
// counts are forwarded exactly, and its stop family is mapped (stop/end_turn/STOP -> the
// client wire's normal stop; length/max_tokens/MAX_TOKENS -> length/max_tokens).

import { isRecord, type Json } from './providers.ts';

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic image source as an OpenAI image_url part, or the raw data URL for Gemini. */
function openAIImagePart(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** One OpenAI image_url part as an Anthropic image block. */
function anthropicImageBlock(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * System messages are joined into one system string; user parts keep their text and images;
 * assistant tool_calls become tool_use blocks; tool messages become tool_result blocks.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const block = anthropicImageBlock(part);
        if (block !== null) blocks.push(block);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Parse a data-URL image into its mime type and base64 data, or null. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One Gemini content part for an OpenAI image_url part. */
function geminiImagePart(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** One Gemini content part for an Anthropic image block. */
function geminiImagePartFromAnthropic(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/**
 * An OpenAI or Anthropic client request as a Gemini generateContent request. `from` names the
 * client wire the body arrived on. The model is not in the body: it goes in the URL path.
 */
export function toGeminiRequest(body: Json, from: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  const toolNames = new Map<string, string>(); // tool_call_id -> function name, for tool results
  let systemText = '';
  let tools: Json[] = [];

  if (from === 'openai') {
    const systemParts: string[] = [];
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          const name = typeof call.function.name === 'string' ? call.function.name : '';
          if (typeof call.id === 'string') toolNames.set(call.id, name);
          parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        const name = typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) ?? '' : '';
        contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: toolResultText(raw.content) } } }] });
        continue;
      }
      if (raw.role !== 'user') continue;
      const content = raw.content;
      if (typeof content === 'string') {
        contents.push({ role: 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url') {
          const p = geminiImagePart(part);
          if (p !== null) parts.push(p);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      const decls = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
      });
      if (decls.length > 0) tools = [{ functionDeclarations: decls }];
    }
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        const content = raw.content;
        const parts: Json[] = [];
        const text = textOf(content);
        if (text !== '') parts.push({ text });
        if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block) || block.type !== 'tool_use') continue;
            const name = typeof block.name === 'string' ? block.name : '';
            if (typeof block.id === 'string') toolNames.set(block.id, name);
            parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      const content = raw.content;
      if (typeof content === 'string') {
        contents.push({ role: 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const p = geminiImagePartFromAnthropic(block);
          if (p !== null) parts.push(p);
        } else if (block.type === 'tool_result') {
          const name = typeof block.tool_use_id === 'string' ? toolNames.get(block.tool_use_id) ?? '' : '';
          parts.push({ functionResponse: { name, response: { result: toolResultText(block.content) } } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      const decls = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
      });
      if (decls.length > 0) tools = [{ functionDeclarations: decls }];
    }
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  const maxTokens = from === 'openai' ? (typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined) : typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  const stops = from === 'openai' ? body.stop : body.stop_sequences;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops.filter((s) => typeof s === 'string');
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (from === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One provider answer, parsed: the whole text, the usage counts and the stop family (0 stop, 1 length, 2 tool, 3 refusal). */
export interface ParsedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

/** Split raw SSE text into its frames' data payloads (one string per event, "" for [DONE] markers handled by callers). */
function sseDataFrames(text: string): string[] {
  const out: string[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** Split raw SSE text into (event, data) pairs, in order. */
function sseEvents(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function parseOpenAI(bodyText: string, stream: boolean): ParsedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text, inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens), stopFamily: stopFamilyOf(choice.finish_reason) };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 0;
  for (const payload of sseDataFrames(bodyText)) {
    if (payload.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stop = stopFamilyOf(choice.finish_reason);
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: stop };
}

function parseAnthropic(bodyText: string, stream: boolean): ParsedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text: textOf(value.content), inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens), stopFamily: stopFamilyOf(value.stop_reason) };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 0;
  for (const { event, data } of sseEvents(bodyText)) {
    let value: unknown;
    try {
      value = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) input = numberOrZero(value.message.usage.input_tokens);
    else if (event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') text += value.delta.text;
    else if (event === 'message_delta') {
      if (isRecord(value.delta) && typeof value.delta.stop_reason === 'string') stop = stopFamilyOf(value.delta.stop_reason);
      if (isRecord(value.usage)) output = numberOrZero(value.usage.output_tokens);
    }
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: stop };
}

function parseGemini(bodyText: string, stream: boolean): ParsedAnswer | null {
  const geminiText = (value: Json): string => {
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  };
  const geminiUsage = (value: Json): { input: number; output: number } => {
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
  };
  const geminiStop = (value: Json): number => {
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    return stopFamilyOf(first.finishReason);
  };
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const usage = geminiUsage(value);
    return { text: geminiText(value), inputTokens: usage.input, outputTokens: usage.output, stopFamily: geminiStop(value) };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 0;
  for (const payload of sseDataFrames(bodyText)) {
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    text += geminiText(value);
    const usage = geminiUsage(value);
    if (usage.input !== 0) input = usage.input;
    if (usage.output !== 0) output = usage.output;
    const s = geminiStop(value);
    if (s !== 0) stop = s;
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: stop };
}

/** Parse one provider's answer (JSON or SSE) into text, usage and stop family; null when unparseable. */
export function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, bodyText: string): ParsedAnswer | null {
  if (wire === 'openai') return parseOpenAI(bodyText, stream);
  if (wire === 'anthropic') return parseAnthropic(bodyText, stream);
  return parseGemini(bodyText, stream);
}

/** One server-sent event line: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** An OpenAI Chat Completions JSON answer from a parsed provider answer. */
export function openAIJsonAnswer(model: string, answer: ParsedAnswer): Json {
  const stop = answer.stopFamily === 1 ? 'length' : answer.stopFamily === 2 ? 'tool_calls' : answer.stopFamily === 3 ? 'content_filter' : 'stop';
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** An OpenAI Chat Completions SSE stream from a parsed provider answer. */
export function openAISseAnswer(model: string, answer: ParsedAnswer): string {
  const stop = answer.stopFamily === 1 ? 'length' : answer.stopFamily === 2 ? 'tool_calls' : answer.stopFamily === 3 ? 'content_filter' : 'stop';
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (let at = 0; at < answer.text.length; at += 13) frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + 13) }, null)));
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic Messages JSON answer from a parsed provider answer. */
export function anthropicJsonAnswer(model: string, answer: ParsedAnswer): Json {
  const stop = answer.stopFamily === 1 ? 'max_tokens' : answer.stopFamily === 2 ? 'tool_use' : answer.stopFamily === 3 ? 'refusal' : 'end_turn';
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An Anthropic Messages SSE stream from a parsed provider answer. */
export function anthropicSseAnswer(model: string, answer: ParsedAnswer): string {
  const stop = answer.stopFamily === 1 ? 'max_tokens' : answer.stopFamily === 2 ? 'tool_use' : answer.stopFamily === 3 ? 'refusal' : 'end_turn';
  const frames: string[] = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += 13) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text.slice(at, at + 13) } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
