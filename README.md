# RAG Evaluator - Setup Instructions

## Files Included:
1. **evaluate_rag.py** - Main evaluation script
2. **requirements.txt** - Python dependencies
3. **.env.example** - Template for API key
4. **questions.csv** - Sample questions (first 3 from your data)
5. **ground_truth.csv** - Sample ground truth answers (first 3)
6. **rag_answers.csv** - Sample RAG answers for testing

## Step-by-Step Setup in VS Code:

### Step 1: Download All Files
Download all 6 files from this chat and save them to your **RAG-EVALUATOR** folder

### Step 2: Create .env File
1. In VS Code, right-click in the RAG-EVALUATOR folder
2. Click "New File"
3. Name it: `.env` (with the dot at the start)
4. Copy your company's API key
5. Add this line to the .env file:
   ```
   ANTHROPIC_API_KEY=your-company-api-key-here
   ```
6. Replace with your actual API key
7. Save the file (Ctrl+S or Cmd+S)

### Step 3: Verify Your File Structure
Your RAG-EVALUATOR folder should now have:
- evaluate_rag.py
- requirements.txt
- .env
- questions.csv
- ground_truth.csv
- rag_answers.csv

### Step 4: Run Test Evaluation
1. Open Terminal in VS Code (Terminal → New Terminal)
2. Make sure you're in the RAG-EVALUATOR folder:
   ```bash
   cd Desktop/rag-evaluator
   ```
3. Run the script:
   ```bash
   python3 evaluate_rag.py
   ```

### Expected Output:
```
Loading CSV files...
✅ Found 3 questions
✅ Found 3 RAG answers
✅ Found 3 ground truth answers

✅ Will evaluate 3 matching questions

Starting evaluation...
[1/3] Evaluating Question 1... ✅ Precision: 1
[2/3] Evaluating Question 2... ✅ Precision: 1
[3/3] Evaluating Question 3... ✅ Precision: 0

✅ Evaluation complete!
Results saved to: precision_scores.csv
Successfully evaluated: 3
Errors: 0
Average Precision: 0.67
```

### Step 5: Check Results
- A new folder called **Evaluation_Runs** will be created automatically
- Inside it, you'll find a file named like: **precision_scores_20240128_143025.csv**
- The timestamp format is: YYYYMMDD_HHMMSS (Year/Month/Day_Hour/Minute/Second)
- Each time you run the script, a new timestamped file is created
- This keeps your results organized and prevents overwriting previous evaluations

## For Full Evaluation (All 239 Questions):

### Folder Structure:
```
RAG-EVALUATOR/
├── .env
├── evaluate_rag.py
├── requirements.txt
├── questions.csv
├── ground_truth.csv
├── rag_answers.csv
├── README.md
└── Evaluation_Runs/          ← Created automatically
    ├── precision_scores_20240128_143025.csv
    ├── precision_scores_20240128_150130.csv
    └── precision_scores_20240129_091545.csv
```

### Once Testing is Successful:

1. **Replace the sample CSV files** with your full data:
   - Delete the sample `questions.csv`, `ground_truth.csv`, and `rag_answers.csv`
   - Add your complete CSV files with the same names
   
2. **Get RAG answers from Kaiya:**
   - Run all 239 questions through Kaiya
   - Export as CSV with columns: "Question Number" and "RAG Answer"
   - Save as `rag_answers.csv` in the RAG-EVALUATOR folder

3. **Run full evaluation:**
   ```bash
   python3 evaluate_rag.py
   ```

## CSV Format Requirements:

All three CSV files must have these column names:

**questions.csv:**
- Column 1: "Question Number"
- Column 2: "Question"

**ground_truth.csv:**
- Column 1: "Question Number"
- Column 2: "Ground Truth answer"

**rag_answers.csv:**
- Column 1: "Question Number"
- Column 2: "RAG Answer"

**Important:** The Question Number must match across all three files!

## Troubleshooting:

**Error: "ANTHROPIC_API_KEY not found"**
- Make sure .env file exists
- Make sure it contains: `ANTHROPIC_API_KEY=your-key`
- No spaces around the = sign

**Error: "FileNotFoundError"**
- Make sure all three CSV files are in the same folder as evaluate_rag.py
- Check file names are exactly: questions.csv, rag_answers.csv, ground_truth.csv

**Error: "column not found"**
- Make sure your CSV headers match exactly:
  - "Question Number" (not "ID" or "Question_Number")
  - "Question" or "Ground Truth answer" or "RAG Answer"

**Error: "No matching question numbers"**
- Make sure the Question Number values match across all three files
- Check for extra spaces or formatting issues

## Cost Estimate:
- Each evaluation costs approximately $0.01-0.02
- For 239 questions: approximately $2.39-$4.78
- Uses Claude Sonnet 4 model

## Need Help?
If you encounter any errors, copy the error message and ask for help!
