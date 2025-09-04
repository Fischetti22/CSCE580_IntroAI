#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Resume Classification System
=====================================

Systematic improvements for cleaner classification:
1. Real data loading from resume files
2. Multiple classification approaches
3. Feature engineering and selection
4. Cross-validation and model evaluation
5. Systematic preprocessing pipeline
6. Better handling of imbalanced data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import joblib
import argparse

# Sklearn imports for comprehensive ML pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

# Multiple classifiers for comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# For handling imbalanced data
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumeFeatureExtractor:
    """Extract meaningful features from resume text for classification"""
    
    def __init__(self):
        self.skill_patterns = self._build_skill_patterns()
        self.experience_patterns = self._build_experience_patterns()
    
    def _build_skill_patterns(self) -> Dict[str, List[str]]:
        """Build comprehensive skill pattern dictionary"""
        return {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 
                'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'web_dev': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
                'django', 'flask', 'spring', 'laravel', 'rest api', 'graphql'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data mining', 'statistics',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
                'keras', 'nlp', 'computer vision'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                'oracle', 'sqlite', 'database design', 'data modeling'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'ci/cd', 'terraform', 'ansible', 'linux', 'bash'
            ],
            'mobile': [
                'android', 'ios', 'flutter', 'react native', 'xamarin',
                'swift', 'objective-c', 'kotlin', 'mobile development'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'jira', 'confluence', 'slack',
                'visual studio', 'intellij', 'eclipse', 'vim'
            ]
        }
    
    def _build_experience_patterns(self) -> Dict[str, List[str]]:
        """Build experience level indicators"""
        return {
            'junior': ['intern', 'entry level', 'junior', 'assistant', 'trainee', 'beginner'],
            'mid': ['developer', 'engineer', 'analyst', '2-5 years', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'architect', 'manager', '5+ years', 'expert'],
            'executive': ['director', 'vp', 'cto', 'ceo', 'head of', 'chief']
        }
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from resume text"""
        text_lower = text.lower()
        features = {}
        
        # Skill counts by category
        for category, skills in self.skill_patterns.items():
            count = sum(1 for skill in skills if skill in text_lower)
            features[f'{category}_skills'] = count
        
        # Experience level indicators
        for level, indicators in self.experience_patterns.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            features[f'{level}_experience'] = count
        
        # General metrics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['unique_words'] = len(set(text.lower().split()))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Education indicators
        education_terms = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'education']
        features['education_mentions'] = sum(1 for term in education_terms if term in text_lower)
        
        # Project indicators
        project_terms = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
        features['project_mentions'] = sum(1 for term in project_terms if term in text_lower)
        
        return features

class EnhancedResumeClassifier:
    """Enhanced resume classification system with multiple models and evaluation"""
    
    def __init__(self):
        self.feature_extractor = ResumeFeatureExtractor()
        self.vectorizers = {}
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def _initialize_models(self) -> Dict[str, object]:
        """Initialize multiple classification models"""
        models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        return models
    
    def _create_labels_from_text(self, df: pd.DataFrame) -> pd.Series:
        """Create labels based on text content analysis (for demonstration)"""
        labels = []
        
        for _, row in df.iterrows():
            text = row['text'].lower()
            
            # Simple rule-based labeling for demonstration
            # In practice, you'd have manually labeled data
            if any(term in text for term in ['data scientist', 'machine learning', 'data analysis', 'statistics']):
                labels.append('Data Scientist')
            elif any(term in text for term in ['software engineer', 'developer', 'programming', 'software development']):
                labels.append('Software Engineer')
            elif any(term in text for term in ['web developer', 'frontend', 'backend', 'full stack']):
                labels.append('Web Developer')
            elif any(term in text for term in ['devops', 'system admin', 'cloud engineer', 'infrastructure']):
                labels.append('DevOps Engineer')
            elif any(term in text for term in ['product manager', 'project manager', 'scrum master']):
                labels.append('Project Manager')
            else:
                labels.append('General')
        
        return pd.Series(labels)
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare features combining text vectorization and extracted features"""
        
        # Text vectorization
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        text_features = tfidf.fit_transform(df['text']).toarray()
        self.vectorizers['tfidf'] = tfidf
        
        # Extracted features
        extracted_features = []
        for _, row in df.iterrows():
            features = self.feature_extractor.extract_features(row['text'])
            extracted_features.append(features)
        
        feature_df = pd.DataFrame(extracted_features)
        
        # Fill NaN values
        feature_df = feature_df.fillna(0)
        
        # Combine features
        combined_features = np.hstack([text_features, feature_df.values])
        
        # Store feature column names for later reference
        text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        self.feature_columns = text_feature_names + list(feature_df.columns)
        
        return combined_features, feature_df
    
    def train_models(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Dict]:
        """Train multiple models and return evaluation results"""
        
        # Create labels if not provided
        if target_column is None or target_column not in df.columns:
            logger.info("Creating labels based on text content analysis...")
            df['predicted_role'] = self._create_labels_from_text(df)
            target_column = 'predicted_role'
        
        # Prepare features
        logger.info("Extracting and preparing features...")
        X, feature_df = self._prepare_features(df)
        y = self.label_encoder.fit_transform(df[target_column])
        
        # Check class distribution
        class_distribution = Counter(df[target_column])
        logger.info(f"Class distribution: {dict(class_distribution)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize models
        models = self._initialize_models()
        results = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Handle class imbalance for some models
                if model_name in ['logistic_regression', 'random_forest', 'svm']:
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=np.unique(y_train), 
                        y=y_train
                    )
                    if hasattr(model, 'class_weight'):
                        model.set_params(class_weight='balanced')
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Classification report
                report = classification_report(
                    y_test, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': report,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Store the best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.models['best'] = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.3f}")
        
        return results, X_test, y_test
    
    def predict_resume_role(self, text: str) -> Tuple[str, float]:
        """Predict role for a single resume"""
        if 'best' not in self.models:
            raise ValueError("No trained model available. Train models first.")
        
        # Create temporary dataframe
        temp_df = pd.DataFrame({'text': [text]})
        
        # Extract features
        X, _ = self._prepare_features(temp_df)
        
        # Predict
        prediction = self.models['best'].predict(X)[0]
        
        # Get probability if available
        confidence = 0.0
        if hasattr(self.models['best'], 'predict_proba'):
            probabilities = self.models['best'].predict_proba(X)[0]
            confidence = max(probabilities)
        
        # Convert back to label
        role = self.label_encoder.inverse_transform([prediction])[0]
        
        return role, confidence
    
    def save_models(self, filepath: Path):
        """Save trained models and components"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to: {filepath}")
    
    def load_models(self, filepath: Path):
        """Load pre-trained models and components"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.label_encoder = model_data['label_encoder']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_columns = model_data['feature_columns']
        
        logger.info(f"Models loaded from: {filepath}")

def plot_results(results: Dict, output_dir: Path):
    """Create visualization plots for model comparison"""
    
    # Model comparison plot
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    ax1.bar(model_names, accuracies)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Cross-validation scores
    ax2.errorbar(model_names, cv_means, yerr=cv_stds, marker='o')
    ax2.set_title('Cross-Validation Scores')
    ax2.set_ylabel('CV Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model comparison plot saved to: {output_dir / 'model_comparison.png'}")

def main():
    """Main function for enhanced resume classification"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Resume Classification System"
    )
    
    parser.add_argument("--resume-data", required=True,
                       help="CSV file with resume text data")
    parser.add_argument("--output-dir", default="classification_results",
                       help="Output directory for results")
    parser.add_argument("--target-column", default=None,
                       help="Column name for target labels (optional)")
    parser.add_argument("--save-models", action="store_true",
                       help="Save trained models")
    parser.add_argument("--test-text", default=None,
                       help="Test text for single prediction")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("ENHANCED RESUME CLASSIFICATION")
    logger.info("=" * 50)
    
    try:
        # Load data
        logger.info(f"Loading resume data from: {args.resume_data}")
        df = pd.read_csv(args.resume_data)
        
        if 'text' not in df.columns:
            raise ValueError("Data must contain a 'text' column with resume content")
        
        logger.info(f"Loaded {len(df)} resume records")
        
        # Initialize classifier
        classifier = EnhancedResumeClassifier()
        
        # Train models
        results, X_test, y_test = classifier.train_models(df, args.target_column)
        
        # Save detailed results
        results_summary = []
        for model_name, result in results.items():
            summary = {
                'model': model_name,
                'accuracy': result['accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'precision': result['classification_report']['weighted avg']['precision'],
                'recall': result['classification_report']['weighted avg']['recall'],
                'f1_score': result['classification_report']['weighted avg']['f1-score']
            }
            results_summary.append(summary)
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(output_dir / 'model_results.csv', index=False)
        
        # Create visualizations
        plot_results(results, output_dir)
        
        # Save models if requested
        if args.save_models:
            model_path = output_dir / 'trained_models.pkl'
            classifier.save_models(model_path)
        
        # Test single prediction if provided
        if args.test_text:
            role, confidence = classifier.predict_resume_role(args.test_text)
            logger.info(f"Predicted role: {role} (confidence: {confidence:.3f})")
        
        logger.info("=" * 50)
        logger.info("CLASSIFICATION COMPLETE!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise

if __name__ == "__main__":
    main()
