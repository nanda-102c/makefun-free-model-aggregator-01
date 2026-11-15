// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Reads the brief (provider directory, quality table, workload manifest), probes every
// provider's catalog route, registers with the env, then answers the client's 20 requests,
// each with the best free model that meets the request's requirements, and exits 0 when the
// env says the workload is done.
//
// Routing: every free candidate that meets the requirements is ranked by the quality table's
// rating, minus a small penalty per call already spent on its provider (so the top providers'
// per-trial budgets are spread instead of the best one being drained), minus a larger penalty
// for providers that failed recently. A 402/429/5xx (or an unreachable provider) fails the
// attempt over to the next candidate; only a successful call answers the request.
//
// Every wire is spoken both ways: an OpenAI or Anthropic client request is translated into
// the provider's own wire (OpenAI passthrough, Anthropic conversion, Gemini
// generateContent/streamGenerateContent), and the provider's answer is parsed and re-emitted
// in the client wire with its text whole, its usage counts exact and its stop family mapped.
// A JSON-mode client request is never sent to an Anthropic-wire provider, which cannot echo
// json_mode. Nothing here reads the wall clock or unseeded randomness, so a replay sees the
// same bytes.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicJsonAnswer,
  anthropicSseAnswer,
  openAIJsonAnswer,
  openAISseAnswer,
  openAIToAnthropicRequest,
  parseUpstream,
  toGeminiRequest,
  type ParsedAnswer,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before it fails over, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Quality penalty, in milli-points, per call already spent on a provider (budget spread). */
const USED_PENALTY_MILLI = 2;
/** Quality penalty, in milli-points, per failure of a provider in the last few requests. */
const FAIL_PENALTY_MILLI = 100;
/** How many recent requests a provider failure keeps penalising it for. */
const FAIL_WINDOW_REQUESTS = 3;
/**
 * Conservative per-provider call caps: the minimum of each provider's per-trial budget range
 * as the kit's env table draws it (program/env/server.mjs: calls_min..calls_max). A provider
 * never gets more than its guaranteed minimum from this server, so its budget is never the
 * reason a later request fails; past the cap it is still usable as a last resort, just ranked
 * below every provider with budget left.
 */
const BUDGET_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** Quality penalty, in milli-points, once a provider has spent its conservative budget cap. */
const OVER_CAP_PENALTY_MILLI = 500;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Calls spent on each provider so far (a refused call counts, as the providers count it). */
const usedByProvider = new Map<string, number>();
/** Request indexes at which each provider failed, for the recent-failure penalty. */
const failsByProvider = new Map<string, number[]>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

/**
 * The free candidates for one request, best first. The score is the quality table's rating
 * minus a spread penalty for calls already spent on the provider and a penalty for recent
 * failures, so the strongest providers are used up to their budgets in turn instead of one
 * being drained. A JSON-mode client request drops Anthropic-wire providers, which cannot
 * carry (or echo) JSON mode.
 */
function candidatesFor(requires: Requirement, clientJsonMode: boolean, requestIndex: number): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientJsonMode && probe.provider.wire === 'anthropic') continue;
    const used = usedByProvider.get(probe.provider.id) ?? 0;
    const cap = BUDGET_CAP[probe.provider.id] ?? 0;
    const recentFails = (failsByProvider.get(probe.provider.id) ?? []).filter((i) => i >= requestIndex - FAIL_WINDOW_REQUESTS).length;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      out.push({ probe, model, rating, score: rating - USED_PENALTY_MILLI * used - (used >= cap ? OVER_CAP_PENALTY_MILLI : 0) - FAIL_PENALTY_MILLI * recentFails });
    }
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

interface RequestContext {
  requires: Requirement;
  weight: 1 | 2;
  clientJsonMode: boolean;
}

/** The requirements, weight and JSON-mode flag of the request now arriving. */
function contextFor(body: Json): RequestContext {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const clientJsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight, clientJsonMode };
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return { requires: requirementsFromBody(body), weight: 1, clientJsonMode };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** The upstream request for one candidate: the client's body in the provider's own wire. */
function upstreamCall(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    // An Anthropic client body is converted to Chat Completions first: the OpenAI parser
    // reads system messages, not a top-level `system` field, and tool/stop shapes differ.
    const upstream: Json = api === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
    if (stream) {
      upstream.stream = true;
      upstream.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(upstream) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = api === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    if (stream) upstream.stream = true;
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(upstream) };
  }
  const encoded = encodeURIComponent(candidate.model.id);
  const path = stream ? `:streamGenerateContent?alt=sse` : `:generateContent`;
  return { url: `${base}/v1beta/models/${encoded}${path}`, headers, body: JSON.stringify(toGeminiRequest(body, api)) };
}

/**
 * One attempt at one candidate: the provider's parsed answer, or null when it refused,
 * failed or answered something unparseable. Every attempt counts against the provider.
 */
async function tryCandidate(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, stream: boolean, requestIndex: number): Promise<ParsedAnswer | null> {
  const providerId = candidate.probe.provider.id;
  usedByProvider.set(providerId, (usedByProvider.get(providerId) ?? 0) + 1);
  const call = upstreamCall(candidate, api, body, stream);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, { method: 'POST', headers: call.headers, body: call.body, signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    log(`${providerId}/${candidate.model.id}: unreachable: ${(err as Error).message}`);
    noteFail(providerId, requestIndex);
    return null;
  }
  const text = await upstream.text();
  if (!upstream.ok) {
    log(`${providerId}/${candidate.model.id}: HTTP ${upstream.status}`);
    noteFail(providerId, requestIndex);
    return null;
  }
  const parsed = parseUpstream(candidate.probe.provider.wire, stream, text);
  if (parsed === null) {
    log(`${providerId}/${candidate.model.id}: unparseable answer`);
    noteFail(providerId, requestIndex);
    return null;
  }
  return parsed;
}

function noteFail(providerId: string, requestIndex: number): void {
  const list = failsByProvider.get(providerId) ?? [];
  list.push(requestIndex);
  failsByProvider.set(providerId, list);
}

/**
 * Serve one client request: walk the ranked candidates until one answers, then emit its
 * answer in the client wire. Returns false when no candidate answered.
 */
async function serve(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic', body: Json): Promise<boolean> {
  const ctx = contextFor(body);
  const requestIndex = manifestIndex - 1;
  const stream = body.stream === true;
  const triedProviders = new Set<string>();
  for (const candidate of candidatesFor(ctx.requires, ctx.clientJsonMode, requestIndex)) {
    // One attempt per provider per request: a provider's refusal is about the provider, not
    // the model, and its other models share the same budget and failure schedule.
    if (triedProviders.has(candidate.probe.provider.id)) continue;
    triedProviders.add(candidate.probe.provider.id);
    const answer = await tryCandidate(candidate, api, body, stream, requestIndex);
    if (answer === null) continue;
    log(`served ${api} request ${requestIndex + 1} with ${candidate.probe.provider.id}/${candidate.model.id} (rating ${candidate.rating})`);
    if (api === 'openai') {
      if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', openAISseAnswer(candidate.model.id, answer));
      else sendJson(res, 200, openAIJsonAnswer(candidate.model.id, answer));
    } else {
      if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicSseAnswer(candidate.model.id, answer));
      else sendJson(res, 200, anthropicJsonAnswer(candidate.model.id, answer));
    }
    return true;
  }
  return false;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (!(await serve(req, res, 'openai', body))) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (!(await serve(req, res, 'anthropic', body))) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: ReturnType<typeof createServer>): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: ReturnType<typeof createServer>): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
