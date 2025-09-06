# Rankers (Consistency Scoring)

This package provides a small factory for prompt–image consistency scoring methods in `optt2i.ranker`.

- clipscore: CLIP/SigLIP-based scalar score
- decomposed_clipscore: per-noun-phrase CLIPScores + average
- vqa: visual question answering wrapper (Hugging Face pipeline)
- dsg: Davidsonian Scene Graph–style Q&A using an OpenAI‑compatible vision model by default

## Quickstart

```python
from optt2i.ranker import create_ranker, score_with_methods

prompt = "a bike lying on the ground, covered in snow"
image_path = "./path/to/image.png"

# Single method
clip = create_ranker("clipscore")
res = clip.score(prompt, image_path)
print(res.global_score, res.details)

# Multiple methods
all_scores = score_with_methods(prompt, image_path, methods=["clipscore", "decomposed_clipscore", "dsg"]) 
for name, out in all_scores.items():
    print(name, out.global_score)
```

## CLI

Run rankers from the command line:

```
python -m optt2i.ranker --images ./qwen_demo.png \
  --prompts "a bike lying on the ground, covered in snow" \
  --methods clipscore,decomposed_clipscore
```

If no `--images` are provided, it tries `./qwen_demo.png`. If no `--prompts` are provided, it uses two sample prompts. Use `--methods` to select from `clipscore`, `decomposed_clipscore`, `vqa`, `vqa_instructblip`, and `dsg`.

### Use an OpenAI-compatible LLM (openai + dotenv)

Enable LLM-based concept extraction and DSG question generation using an OpenAI-compatible API:

```
pip install openai python-dotenv

# Put credentials in .env (dotenv is loaded automatically)
# .env
# OPENAI_API_KEY=sk-...
# OPENAI_API_BASE=https://api.openai.com   # optional for compatible endpoints
# OPENAI_MODEL=gpt-4o-mini                 # optional

python -m optt2i.ranker \
  --images ./qwen_demo.png \
  --prompts "a bike lying on the ground, covered in snow" \
  --methods decomposed_clipscore,dsg --llm-np --llm-dsg
```

Flags:
- `--llm-np`: Force LLM for concept extraction (for `decomposed_clipscore`). If an API key is available, `decomposed_clipscore` will automatically use the LLM by default; this flag is optional.
- `--llm-dsg`: Use the LLM to generate DSG questions (for `dsg`).
- `--llm-base`, `--llm-key`, `--llm-model`: Override env vars.

Environment variables (dotenv supported):
- `OPENAI_API_KEY`: API key (required to enable LLM features)
- `OPENAI_API_BASE`: Base URL (default: `https://api.openai.com`)
- `OPENAI_MODEL`: Chat/vision model name for LLM utilities and DSG VQA (default: `gpt-4o-mini`)
- `VQA_OPENAI_MODEL`: Override only the model used by DSG's OpenAI‑compatible VQA helper

## DSG Ranker

The DSG ranker decomposes a prompt into atomic yes/no questions and answers them against the image using a vision‑capable chat model.

Two generation modes are supported:

- Structured pipeline (default): few‑shot prompts generate tuples, dependencies, and natural‑language questions using native structured outputs. Requires OpenAI‑compatible access.
- Lightweight mode (optional): pass a `question_generator` callable that returns questions directly; useful if you already have a question graph or want a cheaper LLM call.

Basic usage:

```python
from optt2i.ranker import create_ranker

ranker = create_ranker("dsg")
out = ranker.score("a bike on the ground", "./image.png")
print(out.global_score)
```

Inject your own VQA or question generator:

```python
from optt2i.ranker import create_ranker
from optt2i.ranker.llm import get_openai_client, llm_dsg_question_generator
from optt2i.ranker.dsg import OpenAIVisionVQA

# Custom vision model
vqa = OpenAIVisionVQA(model="gpt-4o-mini")

# Lightweight question generator (OpenAI‑compatible LLM)
client = get_openai_client()
qgen = llm_dsg_question_generator(client, model="gpt-4o-mini")

ranker = create_ranker("dsg", vqa_ranker=vqa, question_generator=qgen)
out = ranker.score("a bike on the ground", "./image.png")
```

Notes:
- The ranker caches DSG state per last prompt. Reuse the same instance to score the same prompt over multiple images efficiently. Passing a different prompt regenerates the DSG.
- If providing a custom VQA, implement `batch_vqa(image, questions, binary=True) -> List[str]`.
- Advanced: you can call `optt2i.ranker.dsg.generate_dsg_structured([prompt])` directly to retrieve the intermediate structured outputs.

## Notes

- Models are lazily loaded. Provide local cache or an environment that can download weights.
- Defaults: CLIP/SigLIP `google/siglip-so400m-patch14-384` for CLIPScore; VQA pipeline `dandelin/vilt-b32-finetuned-vqa` for `vqa` method.
- DSG uses an OpenAI‑compatible vision model by default via `OpenAIVisionVQA`; set `OPENAI_MODEL` or `VQA_OPENAI_MODEL` to control it, or inject your own VQA implementation.
- `DecomposedCLIPScoreRanker` uses an LLM extractor by default. To run fully offline, pass your own `phrase_extractor` callable when creating the ranker.
