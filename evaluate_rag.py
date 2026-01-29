"""
RAG Answer Evaluation Script

Evaluates RAG answers against ground truth using Claude API.
Extracts Precision scores and saves results to CSV.
"""

import csv
import os
import sys
import time
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
    print("Please create a .env file with your API key:")
    print("ANTHROPIC_API_KEY=your_api_key_here")
    sys.exit(1)

client = Anthropic(api_key=api_key)

# Evaluation system prompt
EVALUATION_PROMPT = """You are an evaluation assistant. For every message I send that contains this format:
<first-few lines of text> ← Answer A (Ground Truth) *NOTE, this could be a few lines, but usually is contained within "" or quotations
(blank lines)
<final-line text> ← Answer B (Model Output)
…you must automatically evaluate Answer A versus Answer B using the rules below.
You must ALWAYS apply the rules unless asked to reset them.
====================
EVALUATION FRAMEWORK
You must output three binary scores:
Overall Accuracy (0 or 1)
Recall (0 or 1)
Precision (0 or 1)
====================
DEFINITIONS
Overall Accuracy
Score 1 as long as Answer B stays on the same topic and preserves the general directional meaning, even if:
It simplifies the conclusions
It focuses on only one part of Answer A
It generalizes or over-generalizes
It reframes details in a way that appears different
It compresses variability into a single summarized statement
It emphasizes one pattern more heavily than Answer A
Score 0 if Answer B:
• Is fully off-topic, OR
• Explicitly reverses the main conclusion, OR
• Removes or contradicts a critical qualifying condition or documented exception that materially affects the meaning of Answer A.
Mild or moderate contradiction does NOT trigger a 0.
Only a meaningful distortion of the core message (including erasing documented exceptions) qualifies.
Recall
Score 1 as long as Answer B:
Mentions any major component of Answer A's topic, conclusion, or domain
Reflects the general informational intent
Includes at least one meaningful element of A, even if most supporting detail is missing
Recall should NOT be penalized when B:
Omits contrasts
Omits examples
Summarizes heavily
Focuses on only one segment of A
Score 0 if:
• Answer B is essentially unrelated, OR
• It misses critical exceptions or counterexamples that are essential to Answer A's main conclusion, such that the informational intent is no longer preserved.
Precision
Score 1 if Answer B does not introduce clearly false or fabricated statements.
Score 0 if Answer B adds incorrect, fabricated, or contradictory content.
====================
CRITICAL PHILOSOPHY
Accuracy and Recall depend on broad thematic alignment, not detail preservation.
They should be scored 1 if:
The topic is the same
The directional meaning is recognizable, even if condensed
The answer is a plausible high-level summary of A's intent
They should be scored 0 when:
The answer meaningfully changes the topic itself, OR
It asserts a cleanly opposite conclusion, OR
It erases essential constraints, exceptions, or conditions that are central to Answer A's meaning.
====================
FORMATTING INSTRUCTIONS
Output exactly:
<Accuracy Score> 
<Recall Score>
<Precision Score> 
<Accuracy reasoning — one paragraph> 
<Recall reasoning — one paragraph> 
<Precision reasoning — one paragraph>
No titles. No labels.
====================
CONFIRMATION
If you understand these rules, respond with:
"Evaluationmodeactivated."
"""


def load_csv(filepath, expected_columns):
    """Load CSV file and validate columns."""
    data = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if expected columns exist
            if not all(col in reader.fieldnames for col in expected_columns):
                print(f"❌ Error: {filepath} must have columns: {expected_columns}")
                print(f"   Found columns: {reader.fieldnames}")
                sys.exit(1)
            
            for row in reader:
                question_num = row[expected_columns[0]].strip()
                text = row[expected_columns[1]].strip()
                if question_num and text:
                    data[question_num] = text
        
        return data
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        sys.exit(1)


def parse_evaluation_response(response_text):
    """Parse Claude's evaluation response to extract scores and reasoning."""
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    
    try:
        # Extract scores (first 3 lines)
        accuracy_score = int(lines[0])
        recall_score = int(lines[1])
        precision_score = int(lines[2])
        
        # Extract reasoning paragraphs (lines 4, 5, 6)
        accuracy_reasoning = lines[3] if len(lines) > 3 else ""
        recall_reasoning = lines[4] if len(lines) > 4 else ""
        precision_reasoning = lines[5] if len(lines) > 5 else ""
        
        return {
            'accuracy': accuracy_score,
            'recall': recall_score,
            'precision': precision_score,
            'accuracy_reasoning': accuracy_reasoning,
            'recall_reasoning': recall_reasoning,
            'precision_reasoning': precision_reasoning
        }
    except (ValueError, IndexError) as e:
        print(f"⚠️  Warning: Failed to parse response: {e}")
        return None


def initialize_evaluation_mode():
    """
    Send initial 'ready' message to trigger the confirmation response.
    Returns the base messages to include in subsequent requests.
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system=EVALUATION_PROMPT,
            messages=[
                {"role": "user", "content": "ready"}
            ]
        )

        confirmation = response.content[0].text.strip()

        # Verify we got the expected confirmation
        if "Evaluationmodeactivated" in confirmation.replace(" ", ""):
            return [
                {"role": "user", "content": "ready"},
                {"role": "assistant", "content": confirmation}
            ]
        else:
            print(f"⚠️  Unexpected confirmation response: {confirmation}")
            return [
                {"role": "user", "content": "ready"},
                {"role": "assistant", "content": "Evaluationmodeactivated."}
            ]

    except Exception as e:
        print(f"❌ Error initializing evaluation mode: {e}")
        return None


def evaluate_answer(question_num, question, ground_truth, rag_answer, base_messages):
    """Send evaluation request to Claude API with conversation history."""
    # Format the comparison text
    comparison_text = f"""{ground_truth}

{rag_answer}"""

    # Build messages: base conversation + new evaluation request
    messages = base_messages + [
        {"role": "user", "content": comparison_text}
    ]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=EVALUATION_PROMPT,
            messages=messages
        )

        response_text = response.content[0].text
        result = parse_evaluation_response(response_text)

        if result:
            return result
        else:
            return None

    except Exception as e:
        print(f"❌ Error evaluating Question {question_num}: {e}")
        return None


def main():
    print("Loading CSV files...")
    
    # Load all three CSV files with flexible column names
    questions = load_csv('data sources/questions.csv', ['Question Number', 'Question'])
    ground_truths = load_csv('data sources/ground_truth.csv', ['Question Number', 'Ground Truth answer'])
    rag_answers = load_csv('data sources/rag_answers.csv', ['Question Number', 'RAG Answer'])
    
    print(f"✅ Found {len(questions)} questions")
    print(f"✅ Found {len(rag_answers)} RAG answers")
    print(f"✅ Found {len(ground_truths)} ground truth answers")
    
    # Find common question numbers
    common_ids = set(questions.keys()) & set(ground_truths.keys()) & set(rag_answers.keys())
    
    if not common_ids:
        print("❌ Error: No matching question numbers found across all three files")
        sys.exit(1)
    
    common_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)
    print(f"\n✅ Will evaluate {len(common_ids)} matching questions\n")

    # Initialize evaluation mode by triggering the confirmation
    print("Initializing evaluation mode...", end=" ", flush=True)
    base_messages = initialize_evaluation_mode()
    if base_messages is None:
        print("❌ Failed to initialize evaluation mode")
        sys.exit(1)
    print("✅ Evaluationmodeactivated.")

    # Store results
    results = []
    successful = 0
    errors = 0
    precision_scores = []

    print("\nStarting evaluation...")
    print("-" * 60)
    
    for idx, question_num in enumerate(common_ids, 1):
        question = questions[question_num]
        ground_truth = ground_truths[question_num]
        rag_answer = rag_answers[question_num]
        
        print(f"[{idx}/{len(common_ids)}] Evaluating Question {question_num}...", end=" ", flush=True)

        result = evaluate_answer(question_num, question, ground_truth, rag_answer, base_messages)
        
        if result:
            precision = result['precision']
            precision_scores.append(precision)
            successful += 1
            
            results.append({
                'Question Number': question_num,
                'Question': question,
                'Precision_Score': precision,
                'Precision_Reasoning': result['precision_reasoning'],
                'RAG_Answer': rag_answer,
                'Ground_Truth': ground_truth
            })
            
            print(f"✅ Precision: {precision}")
        else:
            errors += 1
            results.append({
                'Question Number': question_num,
                'Question': question,
                'Precision_Score': 'ERROR',
                'Precision_Reasoning': 'Failed to evaluate',
                'RAG_Answer': rag_answer,
                'Ground_Truth': ground_truth
            })
            print("❌ Failed")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Create Evaluation_Runs folder if it doesn't exist
    output_folder = 'Evaluation_Runs'
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_folder, f'precision_scores_{timestamp}.csv')
    
    # Save results
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Question Number', 'Question', 'Precision_Score', 'Precision_Reasoning', 'RAG_Answer', 'Ground_Truth'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Successfully evaluated: {successful}")
    print(f"Errors: {errors}")
    
    if precision_scores:
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"Average Precision: {avg_precision:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
