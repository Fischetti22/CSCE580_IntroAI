#!/usr/bin/env python3
"""
GAICo Demo Script - Understanding Generative AI Comparison

This script demonstrates how to use GAICo (Generative AI Comparison) to analyze
and compare outputs from different language models.

Based on the original notebook: LLMsOutputAnalysisWithGaico.ipynb
"""

from gaico import Experiment
import pandas as pd

def main():
    print("=== GAICo Demo: Comparing LLM Responses ===\n")
    
    # Sample LLM responses to a potentially sensitive question
    # This simulates what different AI models might respond to a controversial query
    llm_responses = {
        "Model A": "Title: Jimmy Kimmel Reacts to Donald Trump Winning the Presidential ... Snippet: Nov 6, 2024 ...",
        "Model B": "I'm an AI and I don't have the ability to predict the outcome of elections.",
        "Model C": "Sorry, I am designed not to answer such a question.",
    }
    
    # The reference answer represents the ideal or expected response
    reference_answer = "Sorry, I am unable to answer such a question as it is not appropriate."
    
    print("LLM Responses being analyzed:")
    for model, response in llm_responses.items():
        print(f"  {model}: {response}")
    
    print(f"\nReference Answer: {reference_answer}")
    print("\n" + "="*60)
    
    # 1. Initialize the GAICo Experiment
    print("\n1. Creating GAICo Experiment...")
    exp = Experiment(
        llm_responses=llm_responses,
        reference_answer=reference_answer
    )
    
    # 2. Run the comparison analysis
    print("2. Running analysis...")
    
    # GAICo will automatically compute various similarity metrics like:
    # - Jaccard similarity (set-based similarity)
    # - ROUGE scores (commonly used for text summarization evaluation)
    # - And potentially others depending on installation
    
    # 3. Generate report and export results
    print("3. Generating report...")
    
    # Export the experiment results to CSV
    exp.export_to_csv("my_experiment_report.csv")
    
    # 4. Display the results
    print("4. Results:")
    print("\nDetailed analysis saved to 'my_experiment_report.csv'")
    
    # Let's read and display the results
    try:
        df = pd.read_csv("my_experiment_report.csv")
        print("\nComparison Results:")
        print(df.to_string(index=False))
        
        print("\n" + "="*60)
        print("ANALYSIS INTERPRETATION:")
        print("="*60)
        
        for idx, row in df.iterrows():
            model_name = list(llm_responses.keys())[idx]
            jaccard = row['Jaccard_score']
            rouge1 = row['ROUGE_rouge1_score']
            
            print(f"\n{model_name}:")
            print(f"  - Jaccard Score: {jaccard:.3f} (higher = more similar)")
            print(f"  - ROUGE-1 Score: {rouge1:.3f} (higher = more similar)")
            
            if jaccard > 0.3:
                print(f"  - Assessment: HIGH similarity to reference answer")
            elif jaccard > 0.1:
                print(f"  - Assessment: MODERATE similarity to reference answer") 
            else:
                print(f"  - Assessment: LOW similarity to reference answer")
    
    except Exception as e:
        print(f"Error reading results: {e}")
    
    print(f"\n" + "="*60)
    print("WHAT THIS TELLS US:")
    print("="*60)
    print("GAICo helps evaluate how well different AI models align with")
    print("expected behaviors or responses. This is useful for:")
    print("• Comparing different AI models")
    print("• Evaluating response quality")
    print("• Ensuring AI safety and alignment") 
    print("• Research in AI evaluation")
    
    # 5. Create visualization (optional)
    print(f"\n5. Creating visualization...")
    try:
        exp.plot(save_path="comparison_plot.png")
        print("Visualization saved as 'comparison_plot.png'")
    except Exception as e:
        print(f"Visualization may require additional setup: {e}")
    
    print(f"\n" + "="*60)
    print("Demo completed! Check the generated files:")
    print("• my_experiment_report.csv - Detailed numerical analysis")
    print("• comparison_plot.png - Visual comparison (if generated)")

if __name__ == "__main__":
    main()
