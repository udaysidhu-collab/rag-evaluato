"""
RAG Evaluator Web UI

A Streamlit app for uploading CSV/JSON files and running RAG evaluations.
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default evaluation system prompt (same as evaluate_rag.py)
DEFAULT_EVALUATION_PROMPT = """You are an evaluation assistant. For every message I send that contains this format:
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


def get_evaluation_prompt():
    """Get the current evaluation prompt from session state or default."""
    if 'evaluation_prompt' not in st.session_state:
        st.session_state.evaluation_prompt = DEFAULT_EVALUATION_PROMPT
    return st.session_state.evaluation_prompt


# ===== Column Detection =====

# Column name variants for flexible matching
COLUMN_VARIANTS = {
    'id': ['question number', 'number', 'id', 'question_id', 'questionnumber', 'q_number', '#', 'qid', 'q_id'],
    'question': ['question', 'query', 'question text', 'questiontext', 'q', 'question_text'],
    'ground_truth': ['ground truth answer', 'ground truth', 'groundtruth', 'expected answer', 'reference', 'truth', 'expected', 'ground_truth', 'groundtruthanswer'],
    'rag_answer': ['rag answer', 'answer', 'response', 'output', 'model answer', 'tellius kaiya answer', 'rag_answer', 'raganswer', 'model_answer']
}

# Partial match keywords (if column contains any of these)
PARTIAL_MATCH_KEYWORDS = {
    'rag_answer': ['kaiya', 'answer', 'response', 'output'],
    'ground_truth': ['truth', 'expected', 'reference'],
    'question': ['question', 'query'],
    'id': ['number', 'id', '#']
}


def find_column(columns, column_type):
    """Find a column matching the given type using flexible matching."""
    if not columns:
        return None

    variants = COLUMN_VARIANTS.get(column_type, [])
    keywords = PARTIAL_MATCH_KEYWORDS.get(column_type, [])

    # First: Try exact match (case-insensitive)
    for col in columns:
        col_lower = col.lower().strip()
        if col_lower in variants:
            return col

    # Second: Try partial match (column contains keyword)
    for col in columns:
        col_lower = col.lower().strip()
        for keyword in keywords:
            if keyword in col_lower:
                return col

    return None


# ===== JSON Conversion Functions (from convert_json_to_csv.py) =====

def normalize_text(text):
    """Normalize text for better matching."""
    text = text.lower().strip()
    text = ' '.join(text.split())
    return text


def calculate_similarity(text1, text2):
    """Calculate similarity between two texts (0.0 to 1.0)."""
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def extract_data_from_json_content(json_content):
    """Extract question and answer from JSON content."""
    try:
        data = json.loads(json_content)

        question = None
        for msg in data.get('messages', []):
            if msg.get('role') == 'human':
                question = msg.get('content', '')
                break

        answer = None
        for msg in data.get('messages', []):
            if msg.get('role') == 'ai':
                answer = msg.get('content', '')
                break

        return question, answer
    except Exception:
        return None, None


def find_best_match(json_question, reference_questions, threshold=0.85):
    """Find the best matching question from reference questions."""
    best_match = None
    best_score = 0
    best_question_text = None

    for ref_question_text, question_num in reference_questions.items():
        similarity = calculate_similarity(json_question, ref_question_text)

        if similarity > best_score:
            best_score = similarity
            best_match = question_num
            best_question_text = ref_question_text

    if best_score >= threshold:
        return best_match, best_score, best_question_text
    else:
        return None, best_score, best_question_text


def convert_json_files_to_df(json_files, questions_df, threshold=0.85, id_col='Question Number', q_col='Question'):
    """Convert uploaded JSON files to a DataFrame with RAG answers."""
    # Build reference questions dict
    reference_questions = {}
    for _, row in questions_df.iterrows():
        q_num = str(row[id_col]).strip()
        q_text = str(row[q_col]).strip()
        if q_num and q_text:
            reference_questions[q_text] = q_num

    results = []
    failed = []
    low_confidence = []

    for json_file in json_files:
        filename = json_file.name
        content = json_file.read().decode('utf-8')
        json_file.seek(0)  # Reset for potential re-read

        json_question, answer = extract_data_from_json_content(content)

        if json_question is None or answer is None:
            failed.append({'filename': filename, 'reason': 'Could not extract data'})
            continue

        question_num, similarity, matched_text = find_best_match(
            json_question, reference_questions, threshold
        )

        if question_num is None:
            failed.append({
                'filename': filename,
                'reason': f'No match (best: {similarity*100:.1f}%)',
                'json_question': json_question[:80]
            })
            continue

        results.append({
            'Question Number': question_num,
            'RAG Answer': answer,
            'similarity': similarity,
            'filename': filename
        })

        if similarity < 0.95:
            low_confidence.append({
                'filename': filename,
                'question_num': question_num,
                'similarity': similarity
            })

    return pd.DataFrame(results), failed, low_confidence


# ===== Evaluation Functions =====

def parse_evaluation_response(response_text):
    """Parse Claude's evaluation response to extract scores and reasoning."""
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    try:
        accuracy_score = int(lines[0])
        recall_score = int(lines[1])
        precision_score = int(lines[2])
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
    except (ValueError, IndexError):
        return None


def initialize_client():
    """Initialize Anthropic client."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def initialize_evaluation_mode(client):
    """Initialize evaluation mode and return base messages."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system=get_evaluation_prompt(),
            messages=[{"role": "user", "content": "ready"}]
        )
        confirmation = response.content[0].text.strip()
        return [
            {"role": "user", "content": "ready"},
            {"role": "assistant", "content": confirmation}
        ]
    except Exception as e:
        st.error(f"Error initializing evaluation mode: {e}")
        return None


def evaluate_single(client, ground_truth, rag_answer, base_messages):
    """Evaluate a single answer pair."""
    comparison_text = f"{ground_truth}\n\n{rag_answer}"
    messages = base_messages + [{"role": "user", "content": comparison_text}]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=get_evaluation_prompt(),
            messages=messages
        )
        return parse_evaluation_response(response.content[0].text)
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return None


def main():
    st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")

    st.title("📊 RAG Evaluator")

    # Check for API key
    client = initialize_client()
    if not client:
        st.error("⚠️ ANTHROPIC_API_KEY not found. Please set it in environment variables.")
        return

    st.success("✅ API key loaded")

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Evaluation", "⚙️ Prompt Settings"])

    # ===== TAB 2: Prompt Settings =====
    with tab2:
        st.header("Evaluation Prompt Settings")
        st.markdown("Customize the evaluation prompt used by Claude to score RAG answers.")

        # Initialize session state if needed
        if 'evaluation_prompt' not in st.session_state:
            st.session_state.evaluation_prompt = DEFAULT_EVALUATION_PROMPT

        # Text area for editing the prompt
        edited_prompt = st.text_area(
            "Evaluation Prompt",
            value=st.session_state.evaluation_prompt,
            height=500,
            help="Edit the system prompt used for evaluation. Changes are saved automatically."
        )

        # Update session state when edited
        if edited_prompt != st.session_state.evaluation_prompt:
            st.session_state.evaluation_prompt = edited_prompt
            st.success("✅ Prompt updated!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.evaluation_prompt = DEFAULT_EVALUATION_PROMPT
                st.rerun()

        with col2:
            # Show character count
            st.metric("Character Count", len(st.session_state.evaluation_prompt))

        st.markdown("---")
        st.subheader("📖 Prompt Guide")
        st.markdown("""
        **Key sections you can customize:**

        - **DEFINITIONS** - Change how Accuracy, Recall, and Precision are defined
        - **CRITICAL PHILOSOPHY** - Adjust the scoring philosophy
        - **FORMATTING INSTRUCTIONS** - Modify the expected output format

        **Tips:**
        - Keep the output format consistent (scores on separate lines, then reasoning)
        - The confirmation response "Evaluationmodeactivated." helps verify the prompt was understood
        - Test changes with a small batch before running full evaluations
        """)

    # ===== TAB 1: Evaluation =====
    with tab1:
        st.markdown("Upload your Questions CSV, Ground Truth CSV, and RAG answers to evaluate precision, recall, and accuracy.")

        # File upload section - Questions and Ground Truth
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Questions CSV")
            st.caption("Required columns: `Question Number`, `Question`")
            questions_file = st.file_uploader("Upload Questions", type=['csv'], key='q')

        with col2:
            st.subheader("Ground Truth CSV")
            st.caption("Required columns: `Question Number`, `Ground Truth answer`")
            ground_truth_file = st.file_uploader("Upload Ground Truth", type=['csv'], key='gt')

        st.markdown("---")

        # RAG Answer input method selection
        st.subheader("RAG Answers")
        rag_input_method = st.radio(
            "Choose RAG answer input format:",
            options=["JSON Files", "CSV File"],
            horizontal=True,
            help="Upload individual JSON files from your RAG system, or a single CSV with all answers"
        )

        json_files = None
        rag_csv_file = None
        threshold = 0.85

        if rag_input_method == "JSON Files":
            st.caption("Upload JSON files from your RAG system")
            json_files = st.file_uploader("Upload JSON Files", type=['json'], key='json', accept_multiple_files=True)
            # Similarity threshold slider (only for JSON)
            threshold = st.slider("Matching Threshold", min_value=0.5, max_value=1.0, value=0.85, step=0.05,
                                 help="Minimum similarity score for matching questions (85% recommended)")
        else:
            st.caption("Required columns: `Question Number`, `RAG Answer` (or similar)")
            rag_csv_file = st.file_uploader("Upload RAG Answers CSV", type=['csv'], key='rag_csv')

        # Check if we have all required files
        has_rag_input = (rag_input_method == "JSON Files" and json_files) or (rag_input_method == "CSV File" and rag_csv_file)

        # Preview uploaded files
        if questions_file and ground_truth_file and has_rag_input:
            try:
                q_df = pd.read_csv(questions_file)
                gt_df = pd.read_csv(ground_truth_file)

                st.markdown("---")
                st.subheader("📋 Data Preview")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Questions:** {len(q_df)} rows")
                    st.dataframe(q_df.head(), use_container_width=True)

                with col2:
                    st.write(f"**Ground Truth:** {len(gt_df)} rows")
                    st.dataframe(gt_df.head(), use_container_width=True)

                # Flexible column detection
                q_id_col = find_column(q_df.columns, 'id')
                q_text_col = find_column(q_df.columns, 'question')
                gt_id_col = find_column(gt_df.columns, 'id')
                gt_text_col = find_column(gt_df.columns, 'ground_truth')

                # Validate columns
                q_valid = q_id_col is not None and q_text_col is not None
                gt_valid = gt_id_col is not None and gt_text_col is not None

                if q_valid:
                    st.success(f"✅ Questions: Using '{q_id_col}' (ID), '{q_text_col}' (Question)")
                else:
                    st.error(f"❌ Questions CSV: Could not find ID or Question column. Found: {list(q_df.columns)}")

                if gt_valid:
                    st.success(f"✅ Ground Truth: Using '{gt_id_col}' (ID), '{gt_text_col}' (Ground Truth)")
                else:
                    st.error(f"❌ Ground Truth CSV: Could not find ID or Ground Truth column. Found: {list(gt_df.columns)}")

                if q_valid and gt_valid:
                    # Process RAG answers based on input method
                    rag_df = None

                    if rag_input_method == "JSON Files":
                        # Convert JSON files to DataFrame
                        st.markdown("---")
                        st.subheader("🔄 JSON Conversion")
                        st.write(f"**JSON Files:** {len(json_files)} files uploaded")

                        with st.spinner("Converting JSON files..."):
                            rag_df, failed, low_confidence = convert_json_files_to_df(
                                json_files, q_df, threshold, id_col=q_id_col, q_col=q_text_col
                            )

                        if len(rag_df) > 0:
                            st.success(f"✅ Successfully matched {len(rag_df)} JSON files")

                            if failed:
                                with st.expander(f"⚠️ {len(failed)} files failed to match"):
                                    for f in failed:
                                        st.write(f"- **{f['filename']}**: {f['reason']}")

                            if low_confidence:
                                with st.expander(f"⚠️ {len(low_confidence)} matches with low confidence (<95%)"):
                                    for lc in low_confidence:
                                        st.write(f"- **{lc['filename']}** → Q{lc['question_num']} ({lc['similarity']*100:.1f}%)")

                            # Show converted RAG answers
                            st.write("**Converted RAG Answers:**")
                            st.dataframe(rag_df[['Question Number', 'RAG Answer', 'similarity']].head(), use_container_width=True)
                        else:
                            st.error("❌ No JSON files could be matched to questions. Check the matching threshold or file format.")

                    else:
                        # Load RAG answers from CSV
                        st.markdown("---")
                        st.subheader("📄 RAG Answers CSV")

                        rag_csv_df = pd.read_csv(rag_csv_file)
                        st.write(f"**RAG Answers:** {len(rag_csv_df)} rows")
                        st.dataframe(rag_csv_df.head(), use_container_width=True)

                        # Flexible column detection for RAG CSV
                        rag_id_col = find_column(rag_csv_df.columns, 'id')
                        rag_text_col = find_column(rag_csv_df.columns, 'rag_answer')

                        if rag_id_col and rag_text_col:
                            st.success(f"✅ RAG Answers: Using '{rag_id_col}' (ID), '{rag_text_col}' (Answer)")
                            # Rename to standard columns
                            rag_df = rag_csv_df[[rag_id_col, rag_text_col]].rename(
                                columns={rag_id_col: 'Question Number', rag_text_col: 'RAG Answer'}
                            )
                            # Convert Question Number to string for matching
                            rag_df['Question Number'] = rag_df['Question Number'].astype(str).str.strip()
                        else:
                            st.error(f"❌ RAG CSV: Could not find ID or Answer column. Found: {list(rag_csv_df.columns)}")

                    # Continue if we have valid RAG data
                    if rag_df is not None and len(rag_df) > 0:
                        # Rename columns for merging
                        q_df_renamed = q_df[[q_id_col, q_text_col]].rename(columns={q_id_col: 'Question Number', q_text_col: 'Question'})
                        gt_df_renamed = gt_df[[gt_id_col, gt_text_col]].rename(columns={gt_id_col: 'Question Number', gt_text_col: 'Ground Truth answer'})

                        # Convert to string for matching
                        q_df_renamed['Question Number'] = q_df_renamed['Question Number'].astype(str).str.strip()
                        gt_df_renamed['Question Number'] = gt_df_renamed['Question Number'].astype(str).str.strip()

                        # Merge all data
                        merged = pd.merge(
                            q_df_renamed,
                            gt_df_renamed,
                            on='Question Number',
                            how='inner'
                        )
                        merged = pd.merge(
                            merged,
                            rag_df[['Question Number', 'RAG Answer']],
                            on='Question Number',
                            how='inner'
                        )

                        st.success(f"✅ Ready to evaluate {len(merged)} questions")

                        st.markdown("---")

                        # Score selection
                        st.subheader("📊 Select Scores to Include")
                        score_options = st.multiselect(
                            "Choose which scores to include in the output:",
                            options=["Accuracy", "Recall", "Precision"],
                            default=["Accuracy", "Recall", "Precision"],
                            help="Select one or more scores to evaluate and include in the CSV output"
                        )

                        if not score_options:
                            st.warning("⚠️ Please select at least one score type")

                        # Run evaluation button
                        if score_options and st.button("🚀 Run Evaluation", type="primary", use_container_width=True):

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Initialize evaluation mode
                            status_text.text("Initializing evaluation mode...")
                            base_messages = initialize_evaluation_mode(client)

                            if base_messages is None:
                                st.error("Failed to initialize evaluation mode")
                                return

                            results = []

                            for idx, row in merged.iterrows():
                                progress = (idx + 1) / len(merged)
                                progress_bar.progress(progress)
                                status_text.text(f"Evaluating question {idx + 1} of {len(merged)}...")

                                result = evaluate_single(
                                    client,
                                    row['Ground Truth answer'],
                                    row['RAG Answer'],
                                    base_messages
                                )

                                if result:
                                    row_data = {
                                        'Question Number': row['Question Number'],
                                        'Question': row['Question'],
                                    }
                                    # Add selected scores
                                    if "Accuracy" in score_options:
                                        row_data['Accuracy_Score'] = result['accuracy']
                                        row_data['Accuracy_Reasoning'] = result['accuracy_reasoning']
                                    if "Recall" in score_options:
                                        row_data['Recall_Score'] = result['recall']
                                        row_data['Recall_Reasoning'] = result['recall_reasoning']
                                    if "Precision" in score_options:
                                        row_data['Precision_Score'] = result['precision']
                                        row_data['Precision_Reasoning'] = result['precision_reasoning']
                                    # Add answers
                                    row_data['RAG_Answer'] = row['RAG Answer']
                                    row_data['Ground_Truth'] = row['Ground Truth answer']
                                    results.append(row_data)
                                else:
                                    row_data = {
                                        'Question Number': row['Question Number'],
                                        'Question': row['Question'],
                                    }
                                    if "Accuracy" in score_options:
                                        row_data['Accuracy_Score'] = 'ERROR'
                                        row_data['Accuracy_Reasoning'] = 'Failed to evaluate'
                                    if "Recall" in score_options:
                                        row_data['Recall_Score'] = 'ERROR'
                                        row_data['Recall_Reasoning'] = 'Failed to evaluate'
                                    if "Precision" in score_options:
                                        row_data['Precision_Score'] = 'ERROR'
                                        row_data['Precision_Reasoning'] = 'Failed to evaluate'
                                    row_data['RAG_Answer'] = row['RAG Answer']
                                    row_data['Ground_Truth'] = row['Ground Truth answer']
                                    results.append(row_data)

                                time.sleep(0.5)  # Rate limiting

                            progress_bar.progress(1.0)
                            status_text.text("Evaluation complete!")

                            # Display results
                            results_df = pd.DataFrame(results)

                            st.markdown("---")
                            st.subheader("📈 Results")

                            # Summary metrics
                            st.subheader("📊 Summary")

                            # Count errors
                            error_count = 0
                            for r in results:
                                if any(r.get(f'{s}_Score') == 'ERROR' for s in score_options):
                                    error_count += 1

                            # Calculate averages for each selected score
                            metrics_cols = st.columns(len(score_options) + 2)
                            metrics_cols[0].metric("Total Evaluated", len(results))

                            col_idx = 1
                            for score_type in score_options:
                                score_key = f'{score_type}_Score'
                                valid_scores = [r[score_key] for r in results if r.get(score_key) != 'ERROR' and r.get(score_key) is not None]
                                if valid_scores:
                                    avg = sum(valid_scores) / len(valid_scores)
                                    metrics_cols[col_idx].metric(f"Avg {score_type}", f"{avg:.2%}")
                                col_idx += 1

                            metrics_cols[col_idx].metric("Errors", error_count)

                            st.dataframe(results_df, use_container_width=True)

                            # Download button
                            csv = results_df.to_csv(index=False)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            score_suffix = "_".join([s.lower() for s in score_options])
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"evaluation_{score_suffix}_{timestamp}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
