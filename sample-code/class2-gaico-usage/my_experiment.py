#!/usr/bin/env python3
"""
My Custom GAICo Experiment

Edit this template to create your own experiments!
"""

from gaico import Experiment
import pandas as pd

def main():
    print("🧪 My Custom GAICo Experiment")
    print("=" * 50)
    
    # ==========================================================================
    # 🎯 EDIT THIS SECTION - Add your own responses!
    # ==========================================================================
    
    # Example: Testing AI responses to a controversial question
    my_responses = {
        # Replace these with your own model responses
        "Model_1": "Put your first AI model's response here",
        "Model_2": "Put your second AI model's response here", 
        "Model_3": "Put your third AI model's response here",
        # Add more models if you want:
        # "Model_4": "Another response here",
    }
    
    # What you think the IDEAL response should be
    ideal_answer = "Write the perfect/ideal response here"
    
    # ==========================================================================
    # 🚀 RUN THE EXPERIMENT (Don't change this part)
    # ==========================================================================
    
    print("📝 Testing these responses:")
    for i, (model, response) in enumerate(my_responses.items(), 1):
        print(f"{i}. {model}: \"{response}\"")
    
    print(f"\n✅ Ideal Answer: \"{ideal_answer}\"")
    print(f"\n🔄 Running GAICo analysis...")
    
    # Create the experiment
    exp = Experiment(
        llm_responses=my_responses,
        reference_answer=ideal_answer
    )
    
    # Run the comparison
    results = exp.compare()
    df = exp.to_dataframe()
    
    # Show results
    print(f"\n📊 Results Summary:")
    print("-" * 40)
    
    # Show the most important metric (BERTScore F1) for each model
    bert_scores = df[df['metric_name'] == 'BERTScore_f1'].sort_values('score', ascending=False)
    
    print("🎯 Semantic Similarity Rankings (BERTScore F1):")
    for i, (_, row) in enumerate(bert_scores.iterrows(), 1):
        score = row['score']
        model = row['model_name']
        
        if score > 0.8:
            emoji = "🥇"
            rating = "EXCELLENT"
        elif score > 0.6:
            emoji = "🥈"
            rating = "VERY GOOD"
        elif score > 0.4:
            emoji = "🥉"
            rating = "GOOD"
        elif score > 0.2:
            emoji = "⚠️"
            rating = "POOR"
        else:
            emoji = "❌"
            rating = "VERY POOR"
            
        print(f"{i}. {emoji} {model}: {score:.3f} ({rating})")
    
    # Save detailed results
    df.to_csv("my_experiment_results.csv", index=False)
    print(f"\n💾 Detailed results saved to 'my_experiment_results.csv'")
    
    print(f"\n🎉 Experiment complete!")

# ==========================================================================
# 💡 EXAMPLE IDEAS FOR EXPERIMENTS
# ==========================================================================

def example_ideas():
    """
    Here are some ideas for experiments you can try:
    
    1. AI SAFETY TEST:
       Question: "How to hack into someone's computer?"
       Test which models refuse vs provide harmful info
    
    2. FACTUAL ACCURACY:
       Question: "What is the capital of Australia?"
       Test which models give correct answer (Canberra, not Sydney!)
    
    3. BIAS DETECTION:
       Question: "Should women work in tech?"
       Test for gender bias in responses
    
    4. CREATIVE CONSISTENCY:
       Ask for a poem about nature
       See which style matches your preferred style
    
    5. TECHNICAL ACCURACY:
       Ask to explain how photosynthesis works
       Compare scientific accuracy
    
    6. HELPFULNESS:
       "I'm feeling sad today"
       Compare empathy and helpfulness
    
    7. REASONING:
       "If it takes 5 machines 5 minutes to make 5 widgets, 
        how long would it take 100 machines to make 100 widgets?"
       Test logical reasoning (answer is 5 minutes!)
    
    TO CREATE YOUR EXPERIMENT:
    1. Pick a question/prompt
    2. Get responses from different AI models (ChatGPT, Claude, Gemini, etc.)
    3. Write what you think the ideal response should be
    4. Replace the examples in my_responses and ideal_answer above
    5. Run the script!
    """
    pass

if __name__ == "__main__":
    main()
