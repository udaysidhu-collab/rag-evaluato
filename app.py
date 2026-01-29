"""
RAG Evaluator Web UI

A Streamlit app for uploading CSV files and running RAG evaluations.
"""

import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Evaluation system prompt (same as evaluate_rag.py)
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
            system=EVALUATION_PROMPT,
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
            system=EVALUATION_PROMPT,
            messages=messages
        )
        return parse_evaluation_response(response.content[0].text)
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return None


def main():
    st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")

    st.title("📊 RAG Evaluator")
    st.markdown("Upload your Ground Truth and RAG Answer CSV files to evaluate precision, recall, and accuracy.")

    # Check for API key
    client = initialize_client()
    if not client:
        st.error("⚠️ ANTHROPIC_API_KEY not found in .env file. Please add your API key.")
        return

    st.success("✅ API key loaded")

    # File upload section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Questions CSV")
        st.caption("Required columns: `Question Number`, `Question`")
        questions_file = st.file_uploader("Upload Questions", type=['csv'], key='q')

    with col2:
        st.subheader("Ground Truth CSV")
        st.caption("Required columns: `Question Number`, `Ground Truth answer`")
        ground_truth_file = st.file_uploader("Upload Ground Truth", type=['csv'], key='gt')

    with col3:
        st.subheader("RAG Answers CSV")
        st.caption("Required columns: `Question Number`, `RAG Answer`")
        rag_file = st.file_uploader("Upload RAG Answers", type=['csv'], key='rag')

    # Preview uploaded files
    if questions_file and ground_truth_file and rag_file:
        try:
            q_df = pd.read_csv(questions_file)
            gt_df = pd.read_csv(ground_truth_file)
            rag_df = pd.read_csv(rag_file)

            st.markdown("---")
            st.subheader("📋 Data Preview")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Questions:** {len(q_df)} rows")
                st.dataframe(q_df.head(), use_container_width=True)

            with col2:
                st.write(f"**Ground Truth:** {len(gt_df)} rows")
                st.dataframe(gt_df.head(), use_container_width=True)

            with col3:
                st.write(f"**RAG Answers:** {len(rag_df)} rows")
                st.dataframe(rag_df.head(), use_container_width=True)

            # Validate columns
            q_valid = 'Question Number' in q_df.columns and 'Question' in q_df.columns
            gt_valid = 'Question Number' in gt_df.columns and 'Ground Truth answer' in gt_df.columns
            rag_valid = 'Question Number' in rag_df.columns and 'RAG Answer' in rag_df.columns

            if not q_valid:
                st.error("❌ Questions CSV must have columns: `Question Number`, `Question`")
            if not gt_valid:
                st.error("❌ Ground Truth CSV must have columns: `Question Number`, `Ground Truth answer`")
            if not rag_valid:
                st.error("❌ RAG Answers CSV must have columns: `Question Number`, `RAG Answer`")

            if q_valid and gt_valid and rag_valid:
                # Merge data
                merged = pd.merge(
                    q_df[['Question Number', 'Question']],
                    gt_df[['Question Number', 'Ground Truth answer']],
                    on='Question Number',
                    how='inner'
                )
                merged = pd.merge(
                    merged,
                    rag_df[['Question Number', 'RAG Answer']],
                    on='Question Number',
                    how='inner'
                )

                st.success(f"✅ Found {len(merged)} matching questions")

                st.markdown("---")

                # Run evaluation button
                if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):

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
                            results.append({
                                'Question Number': row['Question Number'],
                                'Question': row['Question'],
                                'Precision_Score': result['precision'],
                                'Precision_Reasoning': result['precision_reasoning'],
                                'RAG_Answer': row['RAG Answer'],
                                'Ground_Truth': row['Ground Truth answer']
                            })
                        else:
                            results.append({
                                'Question Number': row['Question Number'],
                                'Question': row['Question'],
                                'Precision_Score': 'ERROR',
                                'Precision_Reasoning': 'Failed to evaluate',
                                'RAG_Answer': row['RAG Answer'],
                                'Ground_Truth': row['Ground Truth answer']
                            })

                        time.sleep(0.5)  # Rate limiting

                    progress_bar.progress(1.0)
                    status_text.text("Evaluation complete!")

                    # Display results
                    results_df = pd.DataFrame(results)

                    st.markdown("---")
                    st.subheader("📈 Results")

                    # Summary metrics
                    valid_scores = [r['Precision_Score'] for r in results if r['Precision_Score'] != 'ERROR']
                    if valid_scores:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Evaluated", len(valid_scores))
                        col2.metric("Average Precision", f"{sum(valid_scores)/len(valid_scores):.2%}")
                        col3.metric("Errors", len(results) - len(valid_scores))

                    st.dataframe(results_df, use_container_width=True)

                    # Download button
                    csv = results_df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"precision_scores_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error reading files: {e}")


if __name__ == "__main__":
    main()
