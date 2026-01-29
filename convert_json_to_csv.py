"""
JSON to CSV Converter for RAG Answers (Smart Question Matching)

This script reads JSON files from Kaiya (your RAG system) and matches them
to your questions.csv file using fuzzy text matching, then creates rag_answers.csv

Usage:
1. Put all your JSON files in a folder called "RAG_JSON_Files"
2. Make sure questions.csv exists in the same directory
3. Run: python3 convert_json_to_csv.py
4. It will create rag_answers.csv automatically
"""

import json
import csv
import os
from pathlib import Path
from difflib import SequenceMatcher


def normalize_text(text):
    """
    Normalize text for better matching.
    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation variations
    """
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text


def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts (0.0 to 1.0)
    Uses Python's built-in SequenceMatcher
    """
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def load_questions_csv(questions_file='questions.csv'):
    """
    Load questions from CSV file.
    
    Returns:
    - dict: {question_text: question_number}
    """
    questions = {}
    
    if not os.path.exists(questions_file):
        print(f"❌ Error: {questions_file} not found!")
        print("Make sure questions.csv is in the same folder as this script.")
        return None
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Try different possible column names
            possible_id_cols = ['Question Number', 'ID', 'Question_Number', 'QuestionNumber']
            possible_text_cols = ['Question', 'Question Text', 'QuestionText', 'Text']
            
            # Find which columns exist
            id_col = None
            text_col = None
            
            for col in possible_id_cols:
                if col in reader.fieldnames:
                    id_col = col
                    break
            
            for col in possible_text_cols:
                if col in reader.fieldnames:
                    text_col = col
                    break
            
            if not id_col or not text_col:
                print(f"❌ Error: Could not find required columns in {questions_file}")
                print(f"Found columns: {reader.fieldnames}")
                return None
            
            print(f"✅ Using columns: '{id_col}' and '{text_col}'")
            
            for row in reader:
                question_num = row[id_col].strip()
                question_text = row[text_col].strip()
                if question_num and question_text:
                    questions[question_text] = question_num
        
        print(f"✅ Loaded {len(questions)} questions from {questions_file}")
        return questions
    
    except Exception as e:
        print(f"❌ Error reading {questions_file}: {e}")
        return None


def extract_data_from_json(json_file_path):
    """
    Extract question and answer from Kaiya JSON file.
    
    Returns:
    - question: str
    - answer: str
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract question (human message)
        question = None
        for msg in data.get('messages', []):
            if msg.get('role') == 'human':
                question = msg.get('content', '')
                break
        
        # Extract answer (AI message)
        answer = None
        for msg in data.get('messages', []):
            if msg.get('role') == 'ai':
                answer = msg.get('content', '')
                break
        
        return question, answer
    
    except Exception as e:
        print(f"❌ Error reading {json_file_path}: {e}")
        return None, None


def find_best_match(json_question, reference_questions, threshold=0.85):
    """
    Find the best matching question from reference questions.
    
    Args:
    - json_question: Question text from JSON file
    - reference_questions: Dict of {question_text: question_number}
    - threshold: Minimum similarity score (0.85 = 85% match)
    
    Returns:
    - question_number: str or None
    - similarity: float
    """
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


def convert_json_folder_to_csv(json_folder='RAG_JSON_Files',
                               questions_file='data sources/questions.csv',
                               output_csv='data sources/rag_answers.csv',
                               similarity_threshold=0.85):
    """
    Convert all JSON files in a folder to a single CSV file using smart matching.
    """
    print("=" * 70)
    print("JSON to CSV Converter with Smart Question Matching")
    print("=" * 70)
    print()
    
    # Load reference questions
    reference_questions = load_questions_csv(questions_file)
    if reference_questions is None:
        return
    
    print()
    
    # Check if folder exists
    if not os.path.exists(json_folder):
        print(f"❌ Error: Folder '{json_folder}' not found!")
        print(f"\nPlease create a folder called '{json_folder}' and put your JSON files in it.")
        return
    
    # Get all JSON files
    json_files = list(Path(json_folder).glob('*.json'))
    
    if not json_files:
        print(f"❌ Error: No JSON files found in '{json_folder}'")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Matching threshold: {similarity_threshold * 100}% similarity required")
    print("-" * 70)
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    low_confidence = []
    
    for json_file in json_files:
        filename = json_file.name
        
        # Extract question and answer
        json_question, answer = extract_data_from_json(json_file)
        
        if json_question is None or answer is None:
            print(f"⚠️  Could not extract data from: {filename}")
            failed += 1
            continue
        
        # Find matching question
        question_num, similarity, matched_text = find_best_match(
            json_question, 
            reference_questions, 
            similarity_threshold
        )
        
        if question_num is None:
            print(f"❌ No match found for: {filename}")
            print(f"   Question: {json_question[:80]}...")
            print(f"   Best similarity: {similarity * 100:.1f}% (below {similarity_threshold * 100}% threshold)")
            failed += 1
            continue
        
        results.append({
            'Question Number': question_num,
            'RAG Answer': answer
        })
        
        successful += 1
        
        # Show confidence level
        if similarity < 0.95:
            confidence = "⚠️  LOW"
            low_confidence.append({
                'filename': filename,
                'question_num': question_num,
                'similarity': similarity,
                'json_q': json_question[:60],
                'matched_q': matched_text[:60]
            })
        elif similarity < 0.99:
            confidence = "✓ GOOD"
        else:
            confidence = "✓ PERFECT"
        
        print(f"{confidence} Match: Q{question_num} ({similarity * 100:.1f}%) - {filename[:50]}")
    
    if not results:
        print("\n❌ No files were successfully processed!")
        return
    
    # Sort by question number
    results.sort(key=lambda x: int(x['Question Number']))
    
    # Write to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Question Number', 'RAG Answer'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 70)
    print("✅ Conversion complete!")
    print(f"Successfully matched: {successful} files")
    print(f"Failed to match: {failed} files")
    print(f"Output saved to: {output_csv}")
    
    # Show low confidence matches
    if low_confidence:
        print(f"\n⚠️  {len(low_confidence)} matches had confidence below 95%:")
        print("-" * 70)
        for item in low_confidence[:5]:  # Show first 5
            print(f"Q{item['question_num']} ({item['similarity'] * 100:.1f}%) - {item['filename']}")
            print(f"  JSON:    {item['json_q']}...")
            print(f"  Matched: {item['matched_q']}...")
        if len(low_confidence) > 5:
            print(f"  ... and {len(low_confidence) - 5} more")
        print("\nPlease review these matches to ensure they're correct!")
    
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the matches above (especially any with LOW confidence)")
    print("2. Check rag_answers.csv to verify it looks correct")
    print("3. Run: python3 evaluate_rag.py")

if __name__ == "__main__":
    convert_json_folder_to_csv()
