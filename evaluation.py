#!/usr/bin/env python3

import json
import os
import time
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import torch
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the bitcoin_email_query function from the main module
from ollama_mail_rag import bitcoin_email_query

def load_qna(file_path="qna.json"):
    """Load the question-answer pairs from the JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_answers(qna_data, model_name="mistral-nemo:latest"):
    """Generate answers for each question using the RAG system"""
    print(f"Generating answers using {model_name}...")
    
    # Create the Ollama LLM instance
    llm = ChatOllama(model=model_name)
    
    # Create the system prompt with instructions
    system_template = """You are a helpful assistant that can answer questions about Bitcoin development discussions 
    from the mailing list archives. Use the retrieved information to answer the user's questions.
    If you don't know the answer, say "I don't know."

    Here is relevant information from the Bitcoin email archives:
    {context}
    """
    
    user_template = "{question}"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template)
    ])
    
    # Create the RAG chain
    chain = (
        {"context": bitcoin_email_query, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Generate answers for each question
    generated_answers = []
    
    for i, item in enumerate(tqdm(qna_data)):
        question = item["ques"]
        try:
            # Generate answer
            answer = chain.invoke(question)
            
            # Add to results
            generated_answers.append({
                "question": question,
                "reference_answer": item["ans"],
                "generated_answer": answer
            })
            
            # Brief pause to avoid overloading Ollama
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error generating answer for question {i+1}: {e}")
            # Add error entry
            generated_answers.append({
                "question": question,
                "reference_answer": item["ans"],
                "generated_answer": f"ERROR: {str(e)}"
            })
    
    return generated_answers

def save_results(results, output_file="rag_answers.json"):
    """Save the generated answers to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def open_results(input_file="rag_answers.json"):
    """Load and print the generated answers from a JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    print(f"Results loaded from {input_file}")
    return data

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Download punkt for tokenization
        nltk.download('punkt')
        # Download punkt_tab for sentence tokenization
        nltk.download('punkt_tab')
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Continuing without this data...")

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between reference and hypothesis using simplified approach"""
    try:
        # Simple tokenization by splitting on spaces
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        
        # Calculate BLEU score with smoothing
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_bert_score(references, hypotheses):
    """Calculate BERTScore precision, recall, and F1 scores"""
    try:
        # Filter out empty strings to prevent errors
        valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r and h]
        if not valid_pairs:
            return [], [], []
            
        valid_refs = [r for r, _ in valid_pairs]
        valid_hyps = [h for _, h in valid_pairs]
        
        P, R, F1 = bert_score(valid_hyps, valid_refs, lang="en", verbose=True, batch_size=1)
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        # Return empty lists if there's an error
        return [], [], []

def evaluate_results(results):
    """Evaluate the results using BLEU and BERTScore"""
    # Prepare lists for BLEU scores and BERTScore inputs
    bleu_scores = []
    references = []
    hypotheses = []
    
    # Calculate individual BLEU scores
    for item in results:
        reference = item["reference_answer"]
        hypothesis = item["generated_answer"]
        
        # Skip entries with errors
        if hypothesis.startswith("ERROR:"):
            bleu_scores.append(0.0)
        else:
            bleu_score = calculate_bleu(reference, hypothesis)
            bleu_scores.append(bleu_score)
        
        # Collect references and hypotheses for BERTScore
        references.append(reference)
        hypotheses.append(hypothesis if not hypothesis.startswith("ERROR:") else "")
    
    # Calculate BERTScore
    precision, recall, f1 = calculate_bert_score(references, hypotheses)
    
    # Create evaluation results dictionary
    evaluation = {
        "overall_metrics": {
            "average_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            "average_bertscore_precision": sum(precision) / len(precision) if precision else 0,
            "average_bertscore_recall": sum(recall) / len(recall) if recall else 0,
            "average_bertscore_f1": sum(f1) / len(f1) if f1 else 0
        },
        "individual_metrics": []
    }
    
    # Add individual metrics
    for i, item in enumerate(results):
        eval_item = {
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "generated_answer": item["generated_answer"],
            "bleu_score": bleu_scores[i]
        }
        
        # Add BERTScore if available
        if i < len(precision):
            eval_item.update({
                "bertscore_precision": precision[i],
                "bertscore_recall": recall[i],
                "bertscore_f1": f1[i]
            })
        
        evaluation["individual_metrics"].append(eval_item)
    
    return evaluation

def save_evaluation(evaluation, output_file="rag_evaluation.json"):
    """Save the evaluation results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation saved to {output_file}")

def main(answer_file_generated):
    # Download NLTK data
    download_nltk_data()

    if(answer_file_generated):
        results = open_results()
    else:
        # Load QnA pairs
        qna_data = load_qna()
        print(f"Loaded {len(qna_data)} question-answer pairs")
        
        # Ask user for the Ollama model
        model_name = input("Enter Ollama model to use (default: mistral-nemo:latest): ") or "mistral-nemo:latest"
        
        # Generate answers
        results = generate_answers(qna_data, model_name)
        
        # Save generated answers
        save_results(results)
    
    # Evaluate results
    print("Evaluating results...")
    evaluation = evaluate_results(results)
    
    # Save evaluation
    save_evaluation(evaluation)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average BLEU Score: {evaluation['overall_metrics']['average_bleu']:.4f}")
    print(f"Average BERTScore F1: {evaluation['overall_metrics']['average_bertscore_f1']:.4f}")
    
    # Generate CSV report for easier analysis
    df = pd.DataFrame(evaluation["individual_metrics"])
    df.to_csv("rag_evaluation_report.csv", index=False)
    print("Detailed report saved to rag_evaluation_report.csv")

if __name__ == "__main__":
    answer_file_generated = False # For the first run you dont have the generated answers, so its kept false.
    if(os.path.exists("rag_answers.json")):
        answer_file_generated = True
    main(answer_file_generated) 