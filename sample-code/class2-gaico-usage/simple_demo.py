#!/usr/bin/env python3
"""
Simple GAICo Demo - Based on the working notebook example

This script replicates exactly what the notebook does.
"""

from gaico import Experiment

def main():
    print("=== Simple GAICo Demo ===\n")
    
    # Exactly the same data as in the notebook
    llm_responses = {
        "Model A": "Title: Jimmy Kimmel Reacts to Donald Trump Winning the Presidential ... Snippet: Nov 6, 2024 ...",
        "Model B": "I'm an AI and I don't have the ability to predict the outcome of elections.",
        "Model C": "Sorry, I am designed not to answer such a question.",
    }
    
    reference_answer = "Sorry, I am unable to answer such a question as it is not appropriate."
    
    print("Setting up GAICo experiment...")
    
    # 1. Initialize Experiment (exactly like the notebook)
    exp = Experiment(
        llm_responses=llm_responses,
        reference_answer=reference_answer
    )
    
    print("Experiment created successfully!")
    print(f"Analyzing {len(llm_responses)} model responses...")
    
    # The experiment object automatically computes the metrics when created
    # Let's see what attributes and methods it has
    print("\nExperiment object attributes:")
    attrs = [attr for attr in dir(exp) if not attr.startswith('_')]
    for attr in attrs[:10]:  # Show first 10 attributes
        print(f"  - {attr}")
    if len(attrs) > 10:
        print(f"  ... and {len(attrs) - 10} more")
    
    print("\n" + "="*50)
    print("GAICo is now ready to analyze your model responses!")
    print("The notebook would typically generate visualizations")
    print("and detailed comparisons from this Experiment object.")
    print("="*50)

if __name__ == "__main__":
    main()
