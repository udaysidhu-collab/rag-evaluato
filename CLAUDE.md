# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Evaluator is a Python tool that evaluates Retrieval-Augmented Generation (RAG) system answers against ground truth using Claude AI. It generates precision, recall, and accuracy scores with detailed reasoning.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Stage 1: Convert JSON RAG output to CSV format
python3 convert_json_to_csv.py

# Stage 2: Run evaluation (main entry point)
python3 evaluate_rag.py
```

## Architecture

**Workflow:**
```
RAG_JSON_Files/*.json → convert_json_to_csv.py → rag_answers.csv → evaluate_rag.py → Evaluation_Runs/precision_scores_TIMESTAMP.csv
```

**Two-Stage Pipeline:**
1. `convert_json_to_csv.py` - Converts JSON answers to CSV using fuzzy text matching (85% similarity threshold via `SequenceMatcher`)
2. `evaluate_rag.py` - Evaluates RAG answers against ground truth using Claude Sonnet 4

**Input CSVs:**
- `questions.csv` - Question Number, Question columns
- `ground_truth.csv` - Question Number, Ground Truth columns
- `rag_answers.csv` - Question Number, RAG Answer columns

**Evaluation Scoring (binary 0/1):**
- Precision: Does answer avoid fabricated/false content?
- Recall: Does answer capture major components?
- Accuracy: Does answer stay on topic and preserve meaning?

**API Pattern:**
- Uses conversation history (base_messages) to maintain evaluation context
- System prompt initialized once with confirmation response
- 0.5s rate limiting between API calls
- Max tokens: 100 for init, 2000 for evaluations

## Key Code Locations

- Evaluation system prompt: `evaluate_rag.py:30-99`
- Fuzzy matching threshold: `convert_json_to_csv.py` (85% default, configurable)
- Confidence levels: PERFECT (99-100%), GOOD (95-99%), LOW (85-95%), FAILED (<85%)

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` file. Cost is approximately $0.01-0.02 per question evaluation.
