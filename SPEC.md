# RAG Evaluator Specification

## Overview

RAG Evaluator is a general-purpose Python tool that evaluates Retrieval-Augmented Generation (RAG) system answers against ground truth using Claude AI. It generates precision, recall, and accuracy scores with summarized reasoning.

**Minimum Python Version:** 3.8+

---

## Scoring System

### Metrics
Three binary (0/1) metrics are evaluated for each question:

| Metric | Definition |
|--------|------------|
| **Precision** | Does the answer avoid fabricated/false content? |
| **Recall** | Does the answer capture the major components from ground truth? |
| **Accuracy** | Does the answer stay on topic and preserve meaning? |

### Scoring Rules
- **Binary scoring only**: Each metric is either 0 or 1, no partial credit
- **Semantic equivalence**: Allow reasonable paraphrasing and numerical approximations when comparing against ground truth
- **Length agnostic**: Verbose answers are not penalized; score only on content correctness
- **"I don't know" handling**: If both RAG answer and ground truth indicate no information is available (e.g., "I don't know", "no information found"), all three scores = 1

---

## Evaluation Pipeline

### Two-Stage Workflow
```
RAG_JSON_Files/*.json → convert_json_to_csv.py → rag_answers.csv → evaluate_rag.py → Evaluation_Runs/precision_scores_TIMESTAMP.csv
```

### Stage 1: JSON to CSV Conversion (`convert_json_to_csv.py`)

**Supported JSON Formats** (auto-detected):
1. **Array of objects**: `[{"question": "...", "answer": "..."}]`
2. **Nested with metadata**: `{"results": [{"query": "...", "response": "...", "sources": [...]}]}`
3. **Flat key-value**: `{"Q1": "answer1", "Q2": "answer2"}`

**Matching Strategy:**
- Fuzzy text matching of question content (85% similarity threshold via `SequenceMatcher`)
- Match JSON questions against `questions.csv` using question text, not IDs
- Confidence levels: PERFECT (99-100%), GOOD (95-99%), LOW (85-95%), FAILED (<85%)

**Unmatched Questions:**
- Log a warning with the unmatched question text
- Skip and continue processing other questions

### Stage 2: Evaluation (`evaluate_rag.py`)

**Input Files:**
- `questions.csv` - Question Number, Question columns
- `ground_truth.csv` - Question Number, Ground Truth columns
- `rag_answers.csv` - Question Number, RAG Answer columns

**Column Name Handling:**
- Fuzzy column matching: Accept variations like `question_num`, `QuestionNumber`, `question number` for expected fields
- Case-insensitive, underscore/space tolerant

**Question Set Handling:**
- Evaluate only questions present in ALL three input files (intersection)
- Warn about questions excluded due to missing entries in any file

**Evaluation Order:**
- Questions are evaluated in randomized order to avoid position bias

**Empty Answers:**
- Evaluate as-is (send to Claude for scoring)

---

## Confidence & Escalation

### Low Confidence Detection
Parse Claude's reasoning text for hedging language to detect borderline evaluations.

**Trigger Phrases (minimal set):**
- "borderline"
- "arguably"
- "unclear"
- "could go either way"

### Consensus Voting
When hedging language is detected:
1. Re-run evaluation 3 times with temperature = 0.3
2. Take majority vote for each metric
3. Process silently (no real-time notification to user)
4. Mark consensus-evaluated questions in final output

---

## API Configuration

### Model Selection
- Default: Claude Sonnet
- CLI flag `--model` allows switching (e.g., `--model=haiku`, `--model=opus`)

### Rate Limiting
- Fixed 0.5s delay between API calls
- No adaptive throttling

### Error Recovery
**Checkpoint System:**
- Save progress to temp directory after each question evaluation
- Auto-clean checkpoint on successful completion
- On failure: checkpoint persists for resume

**Resume Behavior:**
- Auto-detect existing checkpoint on startup
- Prompt user: "Resume previous run? [Y/n]"
- If resumed, continue from last successful evaluation

### Token Limits
- Initialization: 100 max tokens
- Evaluations: 2000 max tokens

### Context
- Each question is evaluated independently (no conversation history between evaluations)

---

## Prompt Customization

### Template File
- Location: `evaluation_prompt.txt` in same directory as `evaluate_rag.py`
- Format: Static text (no variable substitution)
- If file doesn't exist, use hardcoded default prompt

---

## Input Validation & Encoding

### File Encoding
- Primary: UTF-8
- Fallback: latin-1 if UTF-8 fails
- Log warnings for encoding conversion issues

### CSV Handling
- Standard Python `csv` module escaping for special characters
- Trust user data for ground truth quality (no length/format validation)

### API Key
- No pre-validation of API key
- Discover issues on first evaluation call

---

## Output

### Results File
- Location: `Evaluation_Runs/precision_scores_TIMESTAMP.csv`
- Format: CSV with standard escaping

### Summary Report
- Included as header rows in the CSV before data
- Example format:
  ```
  #SUMMARY: Total Questions: 50
  #SUMMARY: Precision: 43/50 (86%)
  #SUMMARY: Recall: 38/50 (76%)
  #SUMMARY: Accuracy: 41/50 (82%)
  ```

### Reasoning
- Store 1-2 sentence condensed summary of Claude's reasoning per evaluation
- Full reasoning text not retained

### Debug Log
- Location: `rag_eval.log` in current working directory
- Overwritten each run
- Contains detailed debugging information

---

## User Interface

### Output Mode
- Interactive only (assumes human is watching)
- No batch/CI mode

### Progress Display
- Simple counter: `Evaluating question 15/50...`

### Pre-Run Summary
Before evaluation begins:
1. Display summary of what will be evaluated (question count, files found)
2. Show estimated cost based on question count
3. Prompt for confirmation: "Proceed? [Y/n]"

### Error Messages
- Helpful: Include fix suggestions
- Example: `Error: ground_truth.csv not found. Create this file with columns: Question Number, Ground Truth`

---

## Processing

### Parallelism
- Sequential processing only
- No concurrent API calls

### Caching
- No result caching
- Always re-evaluate on each run

### Question Filtering
- All questions in intersection are evaluated
- No subset selection or sampling

---

## Cost Estimation

- Estimate displayed before run starts
- Based on question count and selected model
- Approximate: $0.01-0.02 per question for Sonnet

---

## Dependencies

Core requirements:
- `anthropic` - Claude API client
- `python-dotenv` - Environment variable loading
- Standard library: `csv`, `json`, `difflib` (SequenceMatcher)

Environment:
- `ANTHROPIC_API_KEY` in `.env` file

---

## File Structure

```
rag-evaluator/
├── evaluate_rag.py           # Main evaluation script
├── convert_json_to_csv.py    # JSON to CSV converter
├── evaluation_prompt.txt     # Custom evaluation prompt (optional)
├── questions.csv             # Input: questions
├── ground_truth.csv          # Input: ground truth answers
├── rag_answers.csv           # Input: RAG system answers
├── rag_eval.log             # Debug log (overwritten each run)
├── RAG_JSON_Files/          # Input: RAG output JSON files
│   └── *.json
├── Evaluation_Runs/         # Output directory
│   └── precision_scores_TIMESTAMP.csv
├── requirements.txt
├── .env                     # API key
├── CLAUDE.md
└── SPEC.md
```
