"""
RAG Evaluator Web UI

A Streamlit app for uploading CSV/JSON files and running RAG evaluations,
plus Ground Truth extraction from documents using multi-LLM consensus.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import re
import time
import tempfile
import pickle
from datetime import datetime
from difflib import SequenceMatcher
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cache file for persisting results across sessions
RESULTS_CACHE_FILE = os.path.join(tempfile.gettempdir(), 'rag_evaluator_results_cache.pkl')


def save_results_cache(results_type, results_data):
    """Save results to cache file for persistence across sessions."""
    try:
        cache_data = {
            'type': results_type,  # 'evaluation' or 'ground_truth'
            'data': results_data,
            'timestamp': datetime.now().isoformat()
        }
        with open(RESULTS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save results cache: {e}")


def load_results_cache():
    """Load cached results if they exist."""
    try:
        if os.path.exists(RESULTS_CACHE_FILE):
            with open(RESULTS_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load results cache: {e}")
    return None


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

# Default extraction prompt
DEFAULT_EXTRACTION_PROMPT = """You are a {INDUSTRY} business intelligence analyst conducting a systematic evidence review. Your role is to extract and synthesize information STRICTLY from the documents provided.

BACKGROUND:
{BACKGROUND}

CRITICAL DOCUMENT SCOPE - READ CAREFULLY:

Answer ONLY using the documents provided in this message
DO NOT use your general knowledge about the subject matter
DO NOT make inferences beyond what is directly stated
DO NOT fill gaps with assumptions or external knowledge

ANSWER GUIDELINES:

If a question cannot be answered from the provided documents, state: "No information found in provided documents"
For partial answers, clearly indicate what information IS present and what is NOT
DO NOT include ANY citations, document names, page numbers, references, or source attributions
If multiple documents address the same question, synthesize the findings

ANSWER LENGTH REQUIREMENT:

When information IS found in the documents, provide a comprehensive, detailed paragraph of at least 4-6 sentences
Include all relevant details, specific criteria, policy nuances, and data points found in the documents
Synthesize information thoroughly to demonstrate comprehensive analysis
Only use the short response "No information found in provided documents" when truly no relevant information exists

YOUR TASK:
Answer each of the following questions using ONLY the information in the provided documents.

MANDATORY OUTPUT FORMAT:
- Provide EXACTLY one answer per question
- Each answer must be on its OWN separate line
- DO NOT add blank lines between answers
- DO NOT add any preamble, summary, or commentary before or after the answers
- The FIRST line of your output should be the answer to question 1
- The LAST line of your output should be the answer to the last question

QUESTIONS:
{QUESTIONS}

Now analyze the provided documents and provide the answers."""

# Synthesis prompt
SYNTHESIS_PROMPT = """You are synthesizing multiple answers into a single comprehensive ground truth answer.

You have been given multiple answers to the same question, collected from different document batches and different AI models. Your task is to synthesize these into ONE final, authoritative answer.

SYNTHESIS GUIDELINES:
1. Combine information from all answers that found relevant data
2. Remove redundancy while keeping all unique details
3. If answers conflict, include both perspectives noting the variation
4. If ALL answers say "No information found", respond with "No information found in provided documents"
5. If SOME answers found information and others didn't, use only the answers that found information
6. Provide a comprehensive, well-structured paragraph (4-6 sentences minimum)
7. DO NOT include any source attributions or citations
8. DO NOT add any preamble - start directly with the synthesized answer

ANSWERS TO SYNTHESIZE:
{ANSWERS}

Provide the single synthesized answer:"""


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
    if columns is None or len(columns) == 0:
        return None

    # Convert to list if it's a pandas Index
    columns = list(columns)

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


# ===== Ground Truth Extraction Functions =====

def load_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n\n'.join(text_parts)
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return '\n\n'.join(text_parts)
        except ImportError:
            st.error("PDF libraries not installed. Run: pip install pdfplumber PyPDF2")
            return None


def load_docx_text(docx_file):
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(docx_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n\n'.join(paragraphs)
    except ImportError:
        st.error("python-docx not installed. Run: pip install python-docx")
        return None


def parse_llm_answers(response_text, num_questions):
    """Parse LLM response into individual answers."""
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    answers = []
    for line in lines:
        # Remove common prefixes like "1.", "1)", "Answer 1:", etc.
        cleaned = re.sub(r'^(\d+[\.\)]\s*|Answer\s*\d+[:\s]*)', '', line, flags=re.IGNORECASE).strip()
        if cleaned:
            answers.append(cleaned)

    # Pad or truncate to expected number
    while len(answers) < num_questions:
        answers.append("No information found in provided documents")

    return answers[:num_questions]


def extract_with_claude(questions, document_text, extraction_prompt, model='claude-sonnet-4-20250514'):
    """Extract answers using Claude API."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None, "ANTHROPIC_API_KEY not found"

    client = Anthropic(api_key=api_key)

    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = extraction_prompt.replace("{QUESTIONS}", questions_text)

    user_message = f"""DOCUMENTS:
{document_text}

{prompt}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": user_message}]
        )
        return parse_llm_answers(response.content[0].text, len(questions)), None
    except Exception as e:
        return None, str(e)


def extract_with_openai(questions, document_text, extraction_prompt, model='gpt-4o'):
    """Extract answers using OpenAI API."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None, "OPENAI_API_KEY not found"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return None, "openai package not installed"

    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = extraction_prompt.replace("{QUESTIONS}", questions_text)

    user_message = f"""DOCUMENTS:
{document_text}

{prompt}"""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": user_message}]
        )
        return parse_llm_answers(response.choices[0].message.content, len(questions)), None
    except Exception as e:
        return None, str(e)


def extract_with_gemini(questions, document_text, extraction_prompt, model='gemini-1.5-pro'):
    """Extract answers using Gemini API."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return None, "GOOGLE_API_KEY not found"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
    except ImportError:
        return None, "google-generativeai package not installed"

    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = extraction_prompt.replace("{QUESTIONS}", questions_text)

    full_prompt = f"""DOCUMENTS:
{document_text}

{prompt}"""

    try:
        response = model_instance.generate_content(full_prompt)
        return parse_llm_answers(response.text, len(questions)), None
    except Exception as e:
        return None, str(e)


def synthesize_answers(question, answers, provider='claude'):
    """Synthesize multiple answers into one."""
    answers_text = "\n\n---\n\n".join([f"Answer {i+1}:\n{a}" for i, a in enumerate(answers)])
    prompt = SYNTHESIS_PROMPT.replace("{ANSWERS}", answers_text)
    full_prompt = f"Question: {question}\n\n{prompt}"

    if provider == 'claude':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return answers[0] if answers else "No information found"
        client = Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text.strip()
        except:
            return answers[0] if answers else "No information found"

    elif provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return answers[0] if answers else "No information found"
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content.strip()
        except:
            return answers[0] if answers else "No information found"

    elif provider == 'gemini':
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return answers[0] if answers else "No information found"
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except:
            return answers[0] if answers else "No information found"

    return answers[0] if answers else "No information found"


# ===== Main App =====

def main():
    st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")

    st.title("📊 RAG Evaluator Suite")

    # Custom CSS for STOP button
    st.markdown("""
    <style>
    div[data-testid="stButton"] button:has-text("⛔ STOP") {
        background-color: #ff0000 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: 2px solid #8B0000 !important;
    }
    div[data-testid="stButton"] button:has-text("⛔ STOP"):hover {
        background-color: #cc0000 !important;
        border: 2px solid #660000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check for API keys
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')

    # Create main tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 RAG Evaluation", "📄 Ground Truth Extraction", "⚙️ Settings"])

    # ===== TAB 3: Settings =====
    with main_tab3:
        st.header("Settings")

        # API Key Status
        st.subheader("🔑 API Key Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            if anthropic_key:
                st.success("✅ Anthropic (Claude)")
            else:
                st.error("❌ Anthropic (Claude)")
        with col2:
            if openai_key:
                st.success("✅ OpenAI")
            else:
                st.warning("⚠️ OpenAI (optional)")
        with col3:
            if google_key:
                st.success("✅ Google (Gemini)")
            else:
                st.warning("⚠️ Google (optional)")

        st.markdown("---")

        # Evaluation Prompt Settings
        st.subheader("📝 Evaluation Prompt")
        st.markdown("Customize the evaluation prompt used by Claude to score RAG answers.")

        if 'evaluation_prompt' not in st.session_state:
            st.session_state.evaluation_prompt = DEFAULT_EVALUATION_PROMPT

        edited_prompt = st.text_area(
            "Evaluation Prompt",
            value=st.session_state.evaluation_prompt,
            height=400,
            help="Edit the system prompt used for evaluation."
        )

        if edited_prompt != st.session_state.evaluation_prompt:
            st.session_state.evaluation_prompt = edited_prompt
            st.success("✅ Prompt updated!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.evaluation_prompt = DEFAULT_EVALUATION_PROMPT
                st.rerun()
        with col2:
            st.metric("Character Count", len(st.session_state.evaluation_prompt))

    # ===== TAB 2: Ground Truth Extraction =====
    with main_tab2:
        st.header("📄 Ground Truth Extraction")
        st.markdown("Extract ground truth answers from documents using multi-LLM consensus (Claude + OpenAI + Gemini).")

        # Check required API key
        if not anthropic_key:
            st.error("⚠️ ANTHROPIC_API_KEY required. Set it in your .env file.")
            return

        # Display cached results if they exist
        if 'last_gt_results' in st.session_state or load_results_cache():
            cached_data = st.session_state.get('last_gt_results') or (
                load_results_cache() if load_results_cache() and load_results_cache().get('type') == 'ground_truth' else None
            )

            if cached_data and cached_data.get('type') != 'evaluation':
                if 'results_df' in cached_data:
                    with st.expander("📋 Last Ground Truth Extraction Results", expanded=False):
                        st.info(f"Showing results from previous run: {cached_data.get('completed', 0)}/{cached_data.get('total_questions', 0)} questions completed")

                        results_df = cached_data['results_df']
                        st.dataframe(results_df, use_container_width=True)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="📥 Download Cached Ground Truth CSV",
                            data=csv,
                            file_name=f"ground_truth_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        st.markdown("---")

        # Configuration Section
        st.subheader("⚙️ Configuration")

        col1, col2 = st.columns(2)
        with col1:
            industry = st.text_input(
                "Industry",
                value="business",
                help="e.g., pharmaceutical, healthcare, finance, technology"
            )
        with col2:
            batch_size = st.number_input(
                "Batch Size (docs per batch)",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of documents to process together in each batch"
            )

        background = st.text_area(
            "Background Context",
            value="Conducting systematic document analysis to extract accurate information.",
            height=100,
            help="Describe the context for the analysis (e.g., product details, analysis goals)"
        )

        st.markdown("---")

        # LLM Selection
        st.subheader("🤖 LLM Providers")

        col1, col2, col3 = st.columns(3)
        with col1:
            use_claude = st.checkbox("Claude", value=True, disabled=not anthropic_key)
            if anthropic_key:
                claude_model = st.selectbox("Claude Model", ["sonnet", "haiku", "opus"], index=0)
            else:
                st.caption("API key missing")
        with col2:
            use_openai = st.checkbox("OpenAI", value=bool(openai_key), disabled=not openai_key)
            if openai_key:
                openai_model = st.selectbox("OpenAI Model", ["gpt4o", "gpt4", "gpt35"], index=0)
            else:
                st.caption("API key missing")
        with col3:
            use_gemini = st.checkbox("Gemini", value=bool(google_key), disabled=not google_key)
            if google_key:
                gemini_model = st.selectbox("Gemini Model", ["pro", "flash"], index=0)
            else:
                st.caption("API key missing")

        synthesis_provider = st.selectbox(
            "Synthesis Provider",
            options=["claude", "openai", "gemini"],
            index=0,
            help="Which LLM to use for synthesizing final answers"
        )

        st.markdown("---")

        # File Upload Section
        st.subheader("📁 Upload Files")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Questions CSV**")
            st.caption("Required columns: `Question Number`, `Question`")
            questions_file = st.file_uploader("Upload Questions CSV", type=['csv'], key='gt_questions')

        with col2:
            st.markdown("**Documents (PDF, DOCX, TXT)**")
            st.caption("Upload all source documents")
            doc_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                key='gt_docs'
            )

        # Preview uploaded files
        if questions_file:
            try:
                q_df = pd.read_csv(questions_file)
                questions_file.seek(0)

                q_id_col = find_column(q_df.columns, 'id')
                q_text_col = find_column(q_df.columns, 'question')

                if q_id_col and q_text_col:
                    st.success(f"✅ Questions: {len(q_df)} questions loaded (using '{q_id_col}' and '{q_text_col}')")
                    with st.expander("Preview Questions"):
                        st.dataframe(q_df.head(10), use_container_width=True)
                else:
                    st.error(f"❌ Could not find required columns. Found: {list(q_df.columns)}")
            except Exception as e:
                st.error(f"Error reading questions file: {e}")

        if doc_files:
            st.success(f"✅ Documents: {len(doc_files)} files uploaded")
            with st.expander("Document List"):
                for f in doc_files:
                    st.write(f"- {f.name}")

        st.markdown("---")

        # Run Extraction
        if questions_file and doc_files and (use_claude or use_openai or use_gemini):
            # Load questions
            q_df = pd.read_csv(questions_file)
            questions_file.seek(0)
            q_id_col = find_column(q_df.columns, 'id')
            q_text_col = find_column(q_df.columns, 'question')

            if q_id_col and q_text_col:
                questions = [(str(row[q_id_col]).strip(), str(row[q_text_col]).strip())
                            for _, row in q_df.iterrows() if row[q_id_col] and row[q_text_col]]

                # Calculate estimates
                num_batches = (len(doc_files) + batch_size - 1) // batch_size
                num_providers = sum([use_claude, use_openai, use_gemini])
                total_api_calls = num_batches * num_providers + len(questions)  # extraction + synthesis

                st.subheader("📊 Pre-Run Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Documents", len(doc_files))
                col2.metric("Batches", num_batches)
                col3.metric("Questions", len(questions))
                col4.metric("API Calls", f"~{total_api_calls}")

                if st.button("🚀 Extract Ground Truth", type="primary", use_container_width=True):

                    # Build extraction prompt
                    extraction_prompt = DEFAULT_EXTRACTION_PROMPT.replace("{INDUSTRY}", industry).replace("{BACKGROUND}", background)

                    # Load all documents
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Loading documents...")
                    documents = []
                    for i, doc_file in enumerate(doc_files):
                        filename = doc_file.name
                        ext = os.path.splitext(filename)[1].lower()

                        if ext == '.pdf':
                            text = load_pdf_text(doc_file)
                        elif ext == '.docx':
                            text = load_docx_text(doc_file)
                        elif ext == '.txt':
                            text = doc_file.read().decode('utf-8', errors='ignore')
                            doc_file.seek(0)
                        else:
                            continue

                        if text:
                            documents.append({'filename': filename, 'text': text})

                        progress_bar.progress((i + 1) / len(doc_files) * 0.1)

                    if not documents:
                        st.error("No documents could be loaded")
                        return

                    status_text.text(f"Loaded {len(documents)} documents")

                    # Create batches
                    batches = []
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        batches.append(batch)

                    # Initialize stop flag
                    st.session_state.stop_requested = False
                    stop_button_placeholder = st.empty()

                    # Process batches
                    question_texts = [q[1] for q in questions]
                    batch_results = {}  # {batch_idx: {provider: [answers]}}

                    total_steps = len(batches) * num_providers
                    current_step = 0

                    for batch_idx, batch in enumerate(batches):
                        # Check if stop was requested
                        if st.session_state.get('stop_requested', False):
                            status_text.text(f"⚠️ Stopped by user. Completed {batch_idx}/{len(batches)} batches.")
                            break

                        # Show STOP button
                        if stop_button_placeholder.button("⛔ STOP", key=f"stop_gt_{batch_idx}",
                                                          help="Stop extraction and synthesize partial results"):
                            st.session_state.stop_requested = True
                            break
                        batch_filenames = [d['filename'] for d in batch]
                        batch_text = "\n\n".join([f"=== DOCUMENT: {d['filename']} ===\n\n{d['text']}" for d in batch])

                        status_text.text(f"Processing batch {batch_idx + 1}/{len(batches)}: {', '.join(batch_filenames)}")
                        batch_results[batch_idx] = {}

                        # Claude
                        if use_claude:
                            model_id = {'sonnet': 'claude-sonnet-4-20250514', 'haiku': 'claude-3-5-haiku-latest', 'opus': 'claude-opus-4-20250219'}[claude_model]
                            answers, error = extract_with_claude(question_texts, batch_text, extraction_prompt, model_id)
                            if answers:
                                batch_results[batch_idx]['claude'] = answers
                            current_step += 1
                            progress_bar.progress(0.1 + (current_step / total_steps) * 0.6)
                            time.sleep(0.5)

                        # OpenAI
                        if use_openai:
                            model_id = {'gpt4o': 'gpt-4o', 'gpt4': 'gpt-4-turbo', 'gpt35': 'gpt-3.5-turbo'}[openai_model]
                            answers, error = extract_with_openai(question_texts, batch_text, extraction_prompt, model_id)
                            if answers:
                                batch_results[batch_idx]['openai'] = answers
                            current_step += 1
                            progress_bar.progress(0.1 + (current_step / total_steps) * 0.6)
                            time.sleep(0.5)

                        # Gemini
                        if use_gemini:
                            model_id = {'pro': 'gemini-1.5-pro', 'flash': 'gemini-1.5-flash'}[gemini_model]
                            answers, error = extract_with_gemini(question_texts, batch_text, extraction_prompt, model_id)
                            if answers:
                                batch_results[batch_idx]['gemini'] = answers
                            current_step += 1
                            progress_bar.progress(0.1 + (current_step / total_steps) * 0.6)
                            time.sleep(0.5)

                    # Synthesis phase
                    status_text.text("Synthesizing answers...")
                    final_results = []

                    for q_idx, (q_num, q_text) in enumerate(questions):
                        # Collect all answers for this question
                        all_answers = []
                        for batch_idx in sorted(batch_results.keys()):
                            for provider_name in ['claude', 'openai', 'gemini']:
                                if provider_name in batch_results[batch_idx]:
                                    answer = batch_results[batch_idx][provider_name][q_idx]
                                    if not answer.startswith("Error:"):
                                        all_answers.append(answer)

                        # Filter out "no information" if we have substantive answers
                        substantive = [a for a in all_answers if "no information found" not in a.lower()]
                        answers_to_synthesize = substantive if substantive else all_answers

                        if not answers_to_synthesize:
                            final_answer = "No information found in provided documents"
                        elif len(answers_to_synthesize) == 1:
                            final_answer = answers_to_synthesize[0]
                        else:
                            final_answer = synthesize_answers(q_text, answers_to_synthesize, synthesis_provider)

                        final_results.append({
                            'Question Number': q_num,
                            'Ground Truth answer': final_answer
                        })

                        progress_bar.progress(0.7 + ((q_idx + 1) / len(questions)) * 0.3)

                    # Clear stop button
                    stop_button_placeholder.empty()

                    progress_bar.progress(1.0)
                    if st.session_state.get('stop_requested', False):
                        status_text.text(f"⚠️ Extraction stopped! Synthesized {len(final_results)}/{len(questions)} questions.")
                    else:
                        status_text.text("Extraction complete!")

                    # Display results
                    st.markdown("---")
                    st.subheader("📋 Results")

                    results_df = pd.DataFrame(final_results)
                    st.dataframe(results_df, use_container_width=True)

                    # Save results to cache
                    cache_data = {
                        'results_df': results_df,
                        'total_questions': len(questions),
                        'completed': len(final_results)
                    }
                    save_results_cache('ground_truth', cache_data)
                    st.session_state.last_gt_results = cache_data

                    # Download button
                    csv = results_df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📥 Download Ground Truth CSV",
                        data=csv,
                        file_name=f"ground_truth_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            if not questions_file:
                st.info("👆 Upload a Questions CSV to get started")
            if not doc_files:
                st.info("👆 Upload source documents (PDF, DOCX, TXT)")
            if not (use_claude or use_openai or use_gemini):
                st.warning("⚠️ Select at least one LLM provider")

    # ===== TAB 1: RAG Evaluation =====
    with main_tab1:
        st.header("📊 RAG Evaluation")

        # Check for API key
        client = initialize_client()
        if not client:
            st.error("⚠️ ANTHROPIC_API_KEY not found. Please set it in environment variables.")
            return

        # Display cached results if they exist
        if 'last_evaluation_results' in st.session_state or load_results_cache():
            cached_data = st.session_state.get('last_evaluation_results') or (
                load_results_cache() if load_results_cache() and load_results_cache().get('type') == 'evaluation' else None
            )

            if cached_data and cached_data.get('type') != 'ground_truth':
                if 'results_df' in cached_data:
                    with st.expander("📋 Last Evaluation Results", expanded=False):
                        st.info(f"Showing results from previous run: {cached_data.get('completed', 0)}/{cached_data.get('total_questions', 0)} questions completed")

                        results_df = cached_data['results_df']
                        score_options = cached_data.get('score_options', ['Accuracy', 'Recall', 'Precision'])

                        # Summary metrics
                        st.subheader("📊 Summary")
                        error_count = sum(1 for _, r in results_df.iterrows()
                                        if any(r.get(f'{s}_Score') == 'ERROR' for s in score_options))

                        metrics_cols = st.columns(len(score_options) + 2)
                        metrics_cols[0].metric("Total Evaluated", len(results_df))

                        col_idx = 1
                        for score_type in score_options:
                            score_key = f'{score_type}_Score'
                            if score_key in results_df.columns:
                                valid_scores = [r for r in results_df[score_key] if r != 'ERROR' and r is not None]
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
                            label="📥 Download Cached Results CSV",
                            data=csv,
                            file_name=f"evaluation_{score_suffix}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

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
                            rag_df = rag_csv_df[[rag_id_col, rag_text_col]].rename(
                                columns={rag_id_col: 'Question Number', rag_text_col: 'RAG Answer'}
                            )
                            rag_df['Question Number'] = rag_df['Question Number'].astype(str).str.strip()
                        else:
                            st.error(f"❌ RAG CSV: Could not find ID or Answer column. Found: {list(rag_csv_df.columns)}")

                    # Continue if we have valid RAG data
                    if rag_df is not None and len(rag_df) > 0:
                        # Rename columns for merging
                        q_df_renamed = q_df[[q_id_col, q_text_col]].rename(columns={q_id_col: 'Question Number', q_text_col: 'Question'})
                        gt_df_renamed = gt_df[[gt_id_col, gt_text_col]].rename(columns={gt_id_col: 'Question Number', gt_text_col: 'Ground Truth answer'})

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
                            # Initialize stop flag
                            st.session_state.stop_requested = False

                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            stop_button_placeholder = st.empty()

                            status_text.text("Initializing evaluation mode...")
                            base_messages = initialize_evaluation_mode(client)

                            if base_messages is None:
                                st.error("Failed to initialize evaluation mode")
                                return

                            results = []

                            for idx, row in merged.iterrows():
                                # Check if stop was requested
                                if st.session_state.get('stop_requested', False):
                                    status_text.text(f"⚠️ Stopped by user. Completed {len(results)}/{len(merged)} questions.")
                                    break

                                # Show STOP button
                                if stop_button_placeholder.button("⛔ STOP", key=f"stop_eval_{idx}",
                                                                  help="Stop evaluation and return partial results"):
                                    st.session_state.stop_requested = True
                                    continue

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
                                    if "Accuracy" in score_options:
                                        row_data['Accuracy_Score'] = result['accuracy']
                                        row_data['Accuracy_Reasoning'] = result['accuracy_reasoning']
                                    if "Recall" in score_options:
                                        row_data['Recall_Score'] = result['recall']
                                        row_data['Recall_Reasoning'] = result['recall_reasoning']
                                    if "Precision" in score_options:
                                        row_data['Precision_Score'] = result['precision']
                                        row_data['Precision_Reasoning'] = result['precision_reasoning']
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

                                time.sleep(0.5)

                            # Clear stop button
                            stop_button_placeholder.empty()

                            progress_bar.progress(1.0)
                            if st.session_state.get('stop_requested', False):
                                status_text.text(f"⚠️ Evaluation stopped! Completed {len(results)}/{len(merged)} questions.")
                            else:
                                status_text.text("Evaluation complete!")

                            # Display results
                            results_df = pd.DataFrame(results)

                            # Save results to cache
                            cache_data = {
                                'results_df': results_df,
                                'score_options': score_options,
                                'total_questions': len(merged),
                                'completed': len(results)
                            }
                            save_results_cache('evaluation', cache_data)
                            st.session_state.last_evaluation_results = cache_data

                            st.markdown("---")
                            st.subheader("📈 Results")

                            # Summary metrics
                            st.subheader("📊 Summary")

                            error_count = 0
                            for r in results:
                                if any(r.get(f'{s}_Score') == 'ERROR' for s in score_options):
                                    error_count += 1

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
