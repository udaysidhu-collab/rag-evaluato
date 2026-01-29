# Smart JSON to CSV Converter - UPDATED WITH FUZZY MATCHING

## 🎯 What Changed?

The converter now uses **intelligent text matching** instead of filename patterns!

### **Old Way (Didn't Work):**
```
query_1_text.json → Question 1  ❌ WRONG!
```
(Filename doesn't equal question ID)

### **New Way (Smart Matching):**
```
JSON Question: "How has Apple's total net sales changed over time?"
               ↓ 99% similarity match
CSV Question:  "How has Apple's total net sales changed over time?"
               ↓
Matched to Question Number: 1  ✅ CORRECT!
```

---

## How It Works

1. **Reads your questions.csv** to get all reference questions
2. **Extracts the question** from each JSON file (`messages[0].content`)
3. **Compares text similarity** using fuzzy matching
4. **Finds the best match** (requires 85%+ similarity by default)
5. **Maps the correct Question Number** to the RAG answer

---

## Step-by-Step Usage

### Step 1: Prepare Files

Make sure these files are in your `RAG-EVALUATOR` folder:
- ✅ `questions.csv` (your reference questions)
- ✅ `convert_json_to_csv.py` (this script)
- ✅ A folder called `RAG_JSON_Files` with all your JSON files

**Folder structure:**
```
RAG-EVALUATOR/
├── questions.csv              ← Your reference questions
├── convert_json_to_csv.py     ← This script
└── RAG_JSON_Files/            ← Create this folder
    ├── query_1.json
    ├── query_2.json
    ├── query_3.json
    └── ... (all JSON files from Kaiya)
```

### Step 2: Run the Converter

In VS Code terminal:
```bash
python3 convert_json_to_csv.py
```

### Step 3: Review the Output

The script will show you each match with confidence levels:

```
======================================================================
JSON to CSV Converter with Smart Question Matching
======================================================================

✅ Using columns: 'Question Number' and 'Question'
✅ Loaded 239 questions from questions.csv

Found 100 JSON files
Matching threshold: 85.0% similarity required
----------------------------------------------------------------------
✓ PERFECT Match: Q1 (100.0%) - query_1.json
✓ PERFECT Match: Q5 (100.0%) - query_2.json
✓ GOOD Match: Q12 (97.3%) - query_3.json
✓ PERFECT Match: Q8 (100.0%) - query_4.json
⚠️  LOW Match: Q23 (89.5%) - query_5.json
...

======================================================================
✅ Conversion complete!
Successfully matched: 98 files
Failed to match: 2 files
Output saved to: rag_answers.csv

⚠️  3 matches had confidence below 95%:
----------------------------------------------------------------------
Q23 (89.5%) - query_5.json
  JSON:    What are the major factors contributing to Apple's gross...
  Matched: What are the major factors contributing to the change in...

Please review these matches to ensure they're correct!
======================================================================
```

---

## Confidence Levels Explained

- **✓ PERFECT (99-100%)**: Exact or nearly exact match - fully trusted
- **✓ GOOD (95-99%)**: Very close match - usually correct
- **⚠️  LOW (85-95%)**: Acceptable match but review recommended
- **❌ FAILED (<85%)**: No good match found - file skipped

---

## What If Questions Don't Match?

### Scenario 1: Low Confidence Matches (85-95%)

The script will list these for you to review. Check if the match makes sense:

```
⚠️  LOW Match: Q23 (89.5%) - query_5.json
  JSON:    What are the major factors contributing to Apple's gross...
  Matched: What are the major factors contributing to the change in...
```

**If it looks correct:** ✅ Keep it
**If it looks wrong:** Manually edit `rag_answers.csv` to fix the Question Number

### Scenario 2: No Match Found (<85%)

File will be skipped with details:

```
❌ No match found for: query_75.json
   Question: What is the weather like today?
   Best similarity: 12.3% (below 85% threshold)
```

**Possible reasons:**
- Question wasn't in your questions.csv
- Question text is completely different
- JSON structure is different

**Solution:** Manually add this answer to rag_answers.csv if needed

---

## Adjusting the Similarity Threshold

Default is 85% (recommended). To change it, edit the script:

**For stricter matching (95%):**
```python
# Line ~230 in convert_json_to_csv.py
convert_json_folder_to_csv(
    similarity_threshold=0.95  # Change from 0.85 to 0.95
)
```

**For looser matching (75%):**
```python
convert_json_folder_to_csv(
    similarity_threshold=0.75  # Use with caution!
)
```

---

## Troubleshooting

### Error: "questions.csv not found"
**Solution:** Make sure questions.csv is in the same folder as the script

### Error: "Could not find required columns"
**Solution:** The script supports these column names:
- ID column: "Question Number", "ID", "Question_ID", "QuestionNumber"
- Text column: "Question", "Question Text", "QuestionText", "Text"

### Too many low confidence matches
**Solutions:**
1. Check if question text in JSON exactly matches questions.csv
2. Lower the threshold to 80-85%
3. Review each match manually

### A JSON file has no match
**Check:**
1. Is the question actually in questions.csv?
2. Is the question text drastically different?
3. View the JSON structure - does it have `messages[0].content`?

---

## Example Output

**Input - questions.csv:**
```csv
Question Number,Question
1,How has Apple's total net sales changed over time?
2,What are the major factors contributing to the change?
3,Has there been any significant change in expenses?
```

**Input - query_1.json:**
```json
{
  "messages": [
    {"role": "human", "content": "How has Apple's total net sales changed over time?"},
    {"role": "ai", "content": "Apple's net sales have fluctuated..."}
  ]
}
```

**Output - rag_answers.csv:**
```csv
Question Number,RAG Answer
1,"Apple's net sales have fluctuated..."
```

The script correctly matched the question text and mapped it to Question Number 1!

---

## Time Saved

**Manual method:** 
- Match each question manually
- Copy/paste 200 times
- Double-check each mapping
- **Total: 8-10 hours**

**Automated method:**
- Drop files in folder
- Run one command
- Review confidence scores
- **Total: 5-10 minutes**

**You just saved 8-10 hours! 🎉**

---

## Next Steps

1. ✅ Run the converter
2. ✅ Review confidence scores
3. ✅ Check rag_answers.csv
4. ✅ Run: `python3 evaluate_rag.py`
5. ✅ Get your precision scores!
