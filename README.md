# Multi-Agent Supervisor System

A supervisor-orchestrated multi-agent system built with **LangGraph**. A supervisor
agent routes each user query to the right specialist — a **research agent** for
live web lookups, or an **analyst agent** for numeric calculations — then returns
a single combined answer.

## Architecture

```
                 ┌──────────────┐
   User query -> │  Supervisor  │
                 └──────┬───────┘
                        │ delegates
           ┌────────────┴────────────┐
           ▼                         ▼
   ┌───────────────┐         ┌───────────────┐
   │ Research Agent │         │ Analyst Agent │
   │ (web search)   │         │ (mean/std/    │
   │                │         │  growth rate) │
   └───────┬────────┘         └───────┬───────┘
           │                          │
           └───────────┬──────────────┘
                        ▼
                  back to Supervisor
                        │
                        ▼
                  Final answer to user
```

- **Supervisor**: decides whether a query needs research, calculation, or a
  direct answer, and hands off control accordingly using LangGraph's
  `Command` primitive.
- **Research Agent**: uses DuckDuckGo search (no paid API key required) to
  answer factual/current-events questions.
- **Analyst Agent**: performs mean, standard deviation, and growth-rate
  calculations on numeric data provided in the conversation.

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Groq](https://groq.com/) — fast LLM inference, running OpenAI's open-weight
  `gpt-oss` model (or Llama 3.3, configurable)
- `duckduckgo-search` — free web search tool (no API key needed)
- `python-dotenv` — environment variable management

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-supervisor.git
cd multi-agent-supervisor

python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GROQ_API_KEY
```

## Run

```bash
python app.py
```

Example session:

```
You: What's the current population of Japan, and what's the growth rate
     if it was 128 million in 2010 and is 123 million now?
```

The supervisor will delegate the population lookup to the research agent,
then delegate the growth-rate math to the analyst agent, and combine both
into a final answer.

## Key Decisions

- **Groq instead of OpenAI**: the original LangGraph supervisor tutorial
  defaults to GPT-4o. This version runs on Groq's inference platform via
  `langchain-groq`, using OpenAI's open-weight `gpt-oss-120b` model by
  default — configurable through the `GROQ_MODEL` environment variable.
  Groq's LPU hardware gives noticeably faster token generation than typical
  API-hosted models, which is a nice property for a multi-agent system that
  makes several sequential LLM calls per query.
- **Free search tool instead of a paid API**: swapped Tavily (requires a paid
  key) for DuckDuckGo search, so the project runs with just one API key.
- **Domain-specific analyst tools**: replaced generic add/multiply/divide
  demo tools with `mean`, `std_dev`, and `growth_rate` — more representative
  of real data-analysis tasks.
- **Manual tool-calling supervisor pattern**: implemented the supervisor with
  explicit handoff tools and a `StateGraph`, rather than the prebuilt
  `langgraph-supervisor` package, for full visibility into how delegation
  works and to align with LangChain's current recommended pattern.

## Possible Extensions

- Add a third worker agent (e.g., a SQL agent or a summarizer agent)
- Add short-term memory with a LangGraph checkpointer
- Wrap this in a Streamlit or FastAPI interface for a live demo
- Add human-in-the-loop approval before the analyst agent runs calculations

## License

MIT
