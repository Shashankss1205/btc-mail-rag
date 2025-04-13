import json
import re
from bs4 import BeautifulSoup
import html

def clean_text(text):
    if not text:
        return ""
    
    # Convert HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')
    
    # Remove Markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove Markdown code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code blocks
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n+', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_qa_pairs():
    # Read input JSON file
    with open('bitcoin_stack_exchange_qa.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    processed_qa = []
    
    for qa in qa_pairs:
        if not qa.get('QUESTION') or not qa.get('ANSWERS'):
            continue
            
        question = qa['QUESTION']
        answers = qa['ANSWERS']
        
        # Get title and body of question
        title = question.get('title', '')
        body = question.get('body', '')
        
        # Clean the text
        clean_title = clean_text(title)
        clean_body = clean_text(body)
        
        # Sort answers by score and get top 2
        sorted_answers = sorted(answers, key=lambda x: x.get('score', 0), reverse=True)
        top_answers = sorted_answers[:2]
        
        # Clean answer bodies
        clean_answers = []
        for ans in top_answers:
            if ans.get('body'):
                clean_answers.append(clean_text(ans['body']))
        
        # Create processed QA entry
        processed_entry = {
            'question': clean_title + clean_body,
            'answers': clean_answers
        }
        
        processed_qa.append(processed_entry)

    # Save processed QA pairs
    with open('processed_bitcoin_qa.json', 'w', encoding='utf-8') as f:
        json.dump(processed_qa, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_qa_pairs()