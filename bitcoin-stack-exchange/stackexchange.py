import requests
import json
import time
from requests.exceptions import RequestException
import os

def load_existing_qa_pairs(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_qa_pair(qa_pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

def fetch_qa_pairs(site="bitcoin", page_size=100, max_pages=10):
    base_url = "https://api.stackexchange.com/2.3"
    output_file = "bitcoin_stack_exchange_qa.json"
    qa_pairs = load_existing_qa_pairs(output_file)
    
    existing_question_ids = {qa['QUESTION']['question_id'] for qa in qa_pairs}
    
    for page in range(1, max_pages + 1):
        print(f"Fetching page {page} of {max_pages}...")
        try:
            params = {
                "site": site,
                "pagesize": page_size,
                "page": page,
                "filter": "withbody",
                "key": "",
                "sort": "votes",
                "order": "desc"
            }
            
            response = requests.get(f"{base_url}/questions", params=params)
            response.raise_for_status()
            questions = response.json().get("items", [])
            
            for question in questions:
                question_id = question.get("question_id")
                
                # Skip if we already have this question
                if question_id in existing_question_ids:
                    continue
                
                answer_params = {
                    "site": site,
                    "filter": "withbody",
                    "sort": "votes",
                    "order": "desc",
                    "key": ""
                }
                
                ans_response = requests.get(f"{base_url}/questions/{question_id}/answers", params=answer_params)
                ans_response.raise_for_status()
                answers = ans_response.json().get("items", [])
                
                if answers:
                    qa_pairs.append({
                        "QUESTION": question,
                        "ANSWERS": answers
                    })
                    # Save after each question is processed
                    save_qa_pair(qa_pairs, output_file)
                    print(f"Saved Q&A pair {len(qa_pairs)} (Question ID: {question_id})")
            
            time.sleep(2)  # API rate limit
                
            if not response.json().get("has_more", False):
                break
                
        except RequestException as e:
            print(f"Error fetching data: {e}")
            continue
    
    return qa_pairs

def main():
    try:
        bitcoin_qa = fetch_qa_pairs(site="bitcoin", max_pages=5)
        print(f"Completed! Total Q&A pairs saved: {len(bitcoin_qa)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()