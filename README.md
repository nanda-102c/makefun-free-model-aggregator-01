# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/nanda-102c/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/nanda-102c/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Node 22 server, stdlib only, serving the OpenAI Chat Completions API and the Anthropic Messages API (JSON and streaming) over ten simulated free-tier providers. It picks the best free model that meets each request's requirements, ranked by the quality table with per-provider budget caps, and fails over to the next candidate on any 402/429/5xx. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

With no trial environment it prints `{"providers":[]}` for `node main.ts --list`. In a trial the env sets `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`; the server probes the providers, registers, serves the workload, and exits 0 when the env says the work is done.

---

<sub>Entered on [make.fun](https://make.fun/?ref=sag0ca1w&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
