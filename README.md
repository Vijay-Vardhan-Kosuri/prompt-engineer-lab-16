# Prompt Quality Evaluation for RAG Climate-Change QA

## Aim
Evaluate prompt quality using systematic metrics (faithfulness & answer relevancy) and discuss ethical considerations in LLM deployment. Two prompt versions are compared on a small climate-change knowledge base.

## Prompt Versions
- **Prompt A (Baseline)**: “Answer the question using the provided context.”
- **Prompt B (Enhanced)**: Adds explicit instructions for faithfulness, conciseness, avoidance of speculation, and resistance to instruction override.

## How to Run
```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install minimal deps (only needed for image generation)
pip install -r requirements.txt

# run the evaluation
python src/prompt_quality_eval.py

# (optional) regenerate the three summary JPGs
python src/generate_output_images.py
```

## Outputs
- Console comparison table of faithfulness / relevancy scores
- Analysis of which prompt performs better and why
- Ethical risk analysis (prompt injection, hallucination, bias) with mitigations
- Prompt-injection simulation demonstrating defence of Prompt B
- `data/results.json` – machine-readable scores
- `output/*.jpg` – three visual summary images

## Notes
- Metrics are a transparent, deterministic approximation of RAGAS (token-overlap based).  
  In a production setting replace the scoring functions with the official `ragas` library and an LLM-as-judge.
- The LLM answers themselves are simulated so the experiment runs offline without API keys.
- Keep the same model & temperature when comparing real prompts.
