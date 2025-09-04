#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Validator for Resume AI Analysis
==============================================

Comprehensive data quality checks and validation:
1. File integrity and readability validation
2. Text extraction quality assessment
3. Data completeness and consistency checks
4. Statistical analysis and outlier detection
5. Systematic reporting and recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter, defaultdict
import json
import argparse
from datetime import datetime

# Text processing
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Comprehensive data quality validation for resume analysis"""
    
    def __init__(self):
        self.validation_results = {}
        self.recommendations = []
        self.quality_score = 0.0
        
        # Quality thresholds
        self.thresholds = {
            'min_text_length': 100,
            'max_text_length': 50000,
            'min_word_count': 50,
            'max_word_count': 10000,
            'min_unique_word_ratio': 0.3,
            'max_special_char_ratio': 0.1,
            'min_sentences': 5,
            'max_extraction_errors': 0.05  # 5% max error rate
        }
    
    def validate_file_integrity(self, file_paths: List[Path]) -> Dict:
        """Validate file accessibility and basic properties"""
        logger.info("Validating file integrity...")
        
        results = {
            'total_files': len(file_paths),
            'accessible_files': 0,
            'inaccessible_files': [],
            'file_types': Counter(),
            'file_sizes': [],
            'empty_files': [],
            'corrupted_files': []
        }
        
        for file_path in file_paths:
            try:
                if not file_path.exists():
                    results['inaccessible_files'].append(str(file_path))
                    continue
                
                # Check file accessibility
                file_size = file_path.stat().st_size
                results['file_sizes'].append(file_size)
                results['file_types'][file_path.suffix.lower()] += 1
                results['accessible_files'] += 1
                
                # Check for empty files
                if file_size == 0:
                    results['empty_files'].append(str(file_path))
                
                # Basic corruption check (for PDF files)
                if file_path.suffix.lower() == '.pdf':
                    try:
                        with open(file_path, 'rb') as f:
                            header = f.read(8)
                            if not header.startswith(b'%PDF'):
                                results['corrupted_files'].append(str(file_path))
                    except Exception:
                        results['corrupted_files'].append(str(file_path))
                        
            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")
                results['inaccessible_files'].append(str(file_path))
        
        # Calculate statistics
        if results['file_sizes']:
            results['avg_file_size'] = np.mean(results['file_sizes'])
            results['median_file_size'] = np.median(results['file_sizes'])
            results['file_size_std'] = np.std(results['file_sizes'])
        
        results['accessibility_rate'] = results['accessible_files'] / results['total_files']
        
        return results
    
    def validate_text_extraction_quality(self, df: pd.DataFrame) -> Dict:
        """Validate quality of extracted text data"""
        logger.info("Validating text extraction quality...")
        
        results = {
            'total_records': len(df),
            'valid_records': 0,
            'empty_texts': 0,
            'short_texts': 0,
            'long_texts': 0,
            'high_special_char_texts': 0,
            'low_diversity_texts': 0,
            'extraction_errors': [],
            'text_lengths': [],
            'word_counts': [],
            'unique_word_ratios': [],
            'special_char_ratios': [],
            'sentence_counts': []
        }
        
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            
            # Basic text metrics
            text_length = len(text)
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            unique_word_ratio = unique_words / word_count if word_count > 0 else 0
            
            # Special character analysis
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
            special_char_ratio = special_chars / text_length if text_length > 0 else 0
            
            # Sentence count
            sentences = len(re.split(r'[.!?]+', text))
            
            # Store metrics
            results['text_lengths'].append(text_length)
            results['word_counts'].append(word_count)
            results['unique_word_ratios'].append(unique_word_ratio)
            results['special_char_ratios'].append(special_char_ratio)
            results['sentence_counts'].append(sentences)
            
            # Validation checks
            if not text.strip():\n                results['empty_texts'] += 1\n                results['extraction_errors'].append({\n                    'index': idx,\n                    'filename': row.get('filename', ''),\n                    'error': 'empty_text'\n                })\n                continue\n            \n            if text_length < self.thresholds['min_text_length']:\n                results['short_texts'] += 1\n                results['extraction_errors'].append({\n                    'index': idx,\n                    'filename': row.get('filename', ''),\n                    'error': 'text_too_short',\n                    'length': text_length\n                })\n            \n            if text_length > self.thresholds['max_text_length']:\n                results['long_texts'] += 1\n                results['extraction_errors'].append({\n                    'index': idx,\n                    'filename': row.get('filename', ''),\n                    'error': 'text_too_long',\n                    'length': text_length\n                })\n            \n            if unique_word_ratio < self.thresholds['min_unique_word_ratio']:\n                results['low_diversity_texts'] += 1\n                results['extraction_errors'].append({\n                    'index': idx,\n                    'filename': row.get('filename', ''),\n                    'error': 'low_word_diversity',\n                    'ratio': unique_word_ratio\n                })\n            \n            if special_char_ratio > self.thresholds['max_special_char_ratio']:\n                results['high_special_char_texts'] += 1\n                results['extraction_errors'].append({\n                    'index': idx,\n                    'filename': row.get('filename', ''),\n                    'error': 'high_special_chars',\n                    'ratio': special_char_ratio\n                })\n            \n            if (text_length >= self.thresholds['min_text_length'] and \n                text_length <= self.thresholds['max_text_length'] and\n                unique_word_ratio >= self.thresholds['min_unique_word_ratio'] and\n                special_char_ratio <= self.thresholds['max_special_char_ratio']):\n                results['valid_records'] += 1\n        \n        # Calculate summary statistics\n        if results['text_lengths']:\n            results['avg_text_length'] = np.mean(results['text_lengths'])\n            results['median_text_length'] = np.median(results['text_lengths'])\n            results['text_length_std'] = np.std(results['text_lengths'])\n            \n            results['avg_word_count'] = np.mean(results['word_counts'])\n            results['median_word_count'] = np.median(results['word_counts'])\n            \n            results['avg_unique_ratio'] = np.mean(results['unique_word_ratios'])\n            results['avg_special_char_ratio'] = np.mean(results['special_char_ratios'])\n            results['avg_sentence_count'] = np.mean(results['sentence_counts'])\n        \n        results['validity_rate'] = results['valid_records'] / results['total_records']\n        results['error_rate'] = len(results['extraction_errors']) / results['total_records']\n        \n        return results\n    \n    def validate_data_completeness(self, df: pd.DataFrame) -> Dict:\n        \"\"\"Validate completeness of data fields\"\"\"\n        logger.info(\"Validating data completeness...\")\n        \n        results = {\n            'total_records': len(df),\n            'required_fields': ['filename', 'text'],\n            'optional_fields': ['label', 'word_count', 'char_count'],\n            'missing_data': {},\n            'duplicate_records': 0,\n            'completeness_score': 0.0\n        }\n        \n        # Check required fields\n        for field in results['required_fields']:\n            if field not in df.columns:\n                results['missing_data'][field] = results['total_records']\n            else:\n                missing_count = df[field].isna().sum()\n                if missing_count > 0:\n                    results['missing_data'][field] = missing_count\n        \n        # Check optional fields\n        for field in results['optional_fields']:\n            if field in df.columns:\n                missing_count = df[field].isna().sum()\n                if missing_count > 0:\n                    results['missing_data'][field] = missing_count\n        \n        # Check for duplicates\n        if 'filename' in df.columns:\n            results['duplicate_records'] = df.duplicated(subset=['filename']).sum()\n        elif 'text' in df.columns:\n            results['duplicate_records'] = df.duplicated(subset=['text']).sum()\n        \n        # Calculate completeness score\n        total_fields = len(results['required_fields']) + len([f for f in results['optional_fields'] if f in df.columns])\n        missing_fields = len(results['missing_data'])\n        results['completeness_score'] = max(0, 1 - (missing_fields / total_fields))\n        \n        return results\n    \n    def detect_outliers_and_anomalies(self, df: pd.DataFrame) -> Dict:\n        \"\"\"Detect statistical outliers and anomalies in the data\"\"\"\n        logger.info(\"Detecting outliers and anomalies...\")\n        \n        results = {\n            'outliers': {},\n            'anomalies': [],\n            'statistical_summary': {}\n        }\n        \n        # Analyze numerical columns\n        numerical_cols = df.select_dtypes(include=[np.number]).columns\n        \n        for col in numerical_cols:\n            if col in df.columns:\n                values = df[col].dropna()\n                if len(values) > 0:\n                    Q1 = values.quantile(0.25)\n                    Q3 = values.quantile(0.75)\n                    IQR = Q3 - Q1\n                    \n                    # IQR method for outlier detection\n                    lower_bound = Q1 - 1.5 * IQR\n                    upper_bound = Q3 + 1.5 * IQR\n                    \n                    outliers = values[(values < lower_bound) | (values > upper_bound)]\n                    \n                    results['outliers'][col] = {\n                        'count': len(outliers),\n                        'percentage': len(outliers) / len(values) * 100,\n                        'values': outliers.tolist()[:10]  # First 10 outliers\n                    }\n                    \n                    results['statistical_summary'][col] = {\n                        'mean': values.mean(),\n                        'median': values.median(),\n                        'std': values.std(),\n                        'min': values.min(),\n                        'max': values.max(),\n                        'q1': Q1,\n                        'q3': Q3\n                    }\n        \n        # Text-specific anomaly detection\n        if 'text' in df.columns:\n            # Detect potentially corrupted text\n            for idx, row in df.iterrows():\n                text = str(row.get('text', ''))\n                \n                # Check for excessive repetition\n                words = text.lower().split()\n                if len(words) > 10:\n                    word_freq = Counter(words)\n                    most_common_word_freq = word_freq.most_common(1)[0][1]\n                    if most_common_word_freq > len(words) * 0.3:  # If one word is >30% of text\n                        results['anomalies'].append({\n                            'index': idx,\n                            'filename': row.get('filename', ''),\n                            'type': 'excessive_repetition',\n                            'most_common_word': word_freq.most_common(1)[0][0],\n                            'frequency': most_common_word_freq\n                        })\n                \n                # Check for encoding issues\n                if re.search(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\xff]', text):\n                    results['anomalies'].append({\n                        'index': idx,\n                        'filename': row.get('filename', ''),\n                        'type': 'encoding_issues'\n                    })\n        \n        return results\n    \n    def calculate_overall_quality_score(self) -> float:\n        \"\"\"Calculate overall data quality score\"\"\"\n        scores = []\n        weights = []\n        \n        # File integrity score (weight: 20%)\n        if 'file_integrity' in self.validation_results:\n            integrity_score = self.validation_results['file_integrity']['accessibility_rate']\n            scores.append(integrity_score)\n            weights.append(0.2)\n        \n        # Text extraction quality score (weight: 40%)\n        if 'text_quality' in self.validation_results:\n            validity_rate = self.validation_results['text_quality']['validity_rate']\n            scores.append(validity_rate)\n            weights.append(0.4)\n        \n        # Data completeness score (weight: 30%)\n        if 'completeness' in self.validation_results:\n            completeness_score = self.validation_results['completeness']['completeness_score']\n            scores.append(completeness_score)\n            weights.append(0.3)\n        \n        # Outlier penalty (weight: 10%)\n        if 'outliers' in self.validation_results:\n            total_outlier_percentage = sum(\n                data['percentage'] for data in self.validation_results['outliers']['outliers'].values()\n            ) / len(self.validation_results['outliers']['outliers']) if self.validation_results['outliers']['outliers'] else 0\n            \n            outlier_score = max(0, 1 - (total_outlier_percentage / 100))\n            scores.append(outlier_score)\n            weights.append(0.1)\n        \n        if scores:\n            self.quality_score = np.average(scores, weights=weights[:len(scores)])\n        \n        return self.quality_score\n    \n    def generate_recommendations(self) -> List[str]:\n        \"\"\"Generate actionable recommendations based on validation results\"\"\"\n        recommendations = []\n        \n        # File integrity recommendations\n        if 'file_integrity' in self.validation_results:\n            integrity = self.validation_results['file_integrity']\n            if integrity['accessibility_rate'] < 0.95:\n                recommendations.append(\n                    f\"Fix file accessibility issues: {len(integrity['inaccessible_files'])} files are inaccessible\"\n                )\n            if integrity['corrupted_files']:\n                recommendations.append(\n                    f\"Replace or repair {len(integrity['corrupted_files'])} corrupted files\"\n                )\n        \n        # Text quality recommendations\n        if 'text_quality' in self.validation_results:\n            quality = self.validation_results['text_quality']\n            if quality['error_rate'] > self.thresholds['max_extraction_errors']:\n                recommendations.append(\n                    f\"Improve text extraction: {quality['error_rate']:.1%} error rate exceeds threshold\"\n                )\n            if quality['empty_texts'] > 0:\n                recommendations.append(\n                    f\"Handle {quality['empty_texts']} empty text extractions\"\n                )\n            if quality['short_texts'] > quality['total_records'] * 0.1:\n                recommendations.append(\n                    \"Consider alternative extraction methods for short text files\"\n                )\n        \n        # Data completeness recommendations\n        if 'completeness' in self.validation_results:\n            completeness = self.validation_results['completeness']\n            if completeness['missing_data']:\n                for field, missing_count in completeness['missing_data'].items():\n                    recommendations.append(\n                        f\"Fill missing data in '{field}': {missing_count} records affected\"\n                    )\n            if completeness['duplicate_records'] > 0:\n                recommendations.append(\n                    f\"Remove or handle {completeness['duplicate_records']} duplicate records\"\n                )\n        \n        # Outlier recommendations\n        if 'outliers' in self.validation_results:\n            outliers = self.validation_results['outliers']['outliers']\n            for field, outlier_data in outliers.items():\n                if outlier_data['percentage'] > 5:  # More than 5% outliers\n                    recommendations.append(\n                        f\"Investigate outliers in '{field}': {outlier_data['count']} values ({outlier_data['percentage']:.1f}%)\"\n                    )\n        \n        self.recommendations = recommendations\n        return recommendations\n    \n    def create_visualizations(self, df: pd.DataFrame, output_dir: Path):\n        \"\"\"Create data quality visualization plots\"\"\"\n        logger.info(\"Creating data quality visualizations...\")\n        \n        # Set up the plotting style\n        plt.style.use('default')\n        sns.set_palette(\"husl\")\n        \n        # Text length distribution\n        if 'text' in df.columns:\n            text_lengths = [len(str(text)) for text in df['text']]\n            \n            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))\n            \n            # Text length histogram\n            ax1.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')\n            ax1.set_title('Distribution of Text Lengths')\n            ax1.set_xlabel('Text Length (characters)')\n            ax1.set_ylabel('Frequency')\n            ax1.axvline(self.thresholds['min_text_length'], color='red', linestyle='--', label='Min threshold')\n            ax1.axvline(self.thresholds['max_text_length'], color='red', linestyle='--', label='Max threshold')\n            ax1.legend()\n            \n            # Word count distribution\n            word_counts = [len(str(text).split()) for text in df['text']]\n            ax2.hist(word_counts, bins=50, alpha=0.7, edgecolor='black', color='orange')\n            ax2.set_title('Distribution of Word Counts')\n            ax2.set_xlabel('Word Count')\n            ax2.set_ylabel('Frequency')\n            ax2.axvline(self.thresholds['min_word_count'], color='red', linestyle='--', label='Min threshold')\n            ax2.legend()\n            \n            # Box plot for text lengths\n            ax3.boxplot(text_lengths)\n            ax3.set_title('Text Length Box Plot')\n            ax3.set_ylabel('Text Length (characters)')\n            \n            # Quality score by label (if available)\n            if 'label' in df.columns:\n                label_quality = df.groupby('label').apply(\n                    lambda x: [len(str(text)) for text in x['text']]\n                ).to_dict()\n                \n                ax4.boxplot(list(label_quality.values()), labels=list(label_quality.keys()))\n                ax4.set_title('Text Length by Label')\n                ax4.set_ylabel('Text Length (characters)')\n                ax4.tick_params(axis='x', rotation=45)\n            else:\n                ax4.text(0.5, 0.5, 'No label data available', ha='center', va='center', transform=ax4.transAxes)\n                ax4.set_title('Label Analysis (N/A)')\n            \n            plt.tight_layout()\n            plt.savefig(output_dir / 'data_quality_distributions.png', dpi=300, bbox_inches='tight')\n            plt.close()\n        \n        # Quality metrics summary\n        if self.validation_results:\n            metrics = []\n            values = []\n            \n            if 'file_integrity' in self.validation_results:\n                metrics.append('File\\nAccessibility')\n                values.append(self.validation_results['file_integrity']['accessibility_rate'])\n            \n            if 'text_quality' in self.validation_results:\n                metrics.append('Text\\nValidity')\n                values.append(self.validation_results['text_quality']['validity_rate'])\n            \n            if 'completeness' in self.validation_results:\n                metrics.append('Data\\nCompleteness')\n                values.append(self.validation_results['completeness']['completeness_score'])\n            \n            metrics.append('Overall\\nQuality')\n            values.append(self.quality_score)\n            \n            # Create quality metrics bar chart\n            fig, ax = plt.subplots(figsize=(10, 6))\n            bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])\n            ax.set_title('Data Quality Metrics', fontsize=16, fontweight='bold')\n            ax.set_ylabel('Score (0-1)', fontsize=12)\n            ax.set_ylim(0, 1.1)\n            \n            # Add value labels on bars\n            for bar, value in zip(bars, values):\n                height = bar.get_height()\n                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')\n            \n            # Add quality threshold line\n            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good Quality Threshold')\n            ax.legend()\n            \n            plt.tight_layout()\n            plt.savefig(output_dir / 'quality_metrics_summary.png', dpi=300, bbox_inches='tight')\n            plt.close()\n    \n    def generate_report(self, output_dir: Path) -> Dict:\n        \"\"\"Generate comprehensive data quality report\"\"\"\n        logger.info(\"Generating data quality report...\")\n        \n        report = {\n            'timestamp': datetime.now().isoformat(),\n            'overall_quality_score': self.quality_score,\n            'validation_results': self.validation_results,\n            'recommendations': self.recommendations,\n            'summary': {\n                'total_files_analyzed': 0,\n                'valid_records': 0,\n                'error_count': 0,\n                'quality_grade': 'F'\n            }\n        }\n        \n        # Calculate summary statistics\n        if 'file_integrity' in self.validation_results:\n            report['summary']['total_files_analyzed'] = self.validation_results['file_integrity']['total_files']\n        \n        if 'text_quality' in self.validation_results:\n            report['summary']['valid_records'] = self.validation_results['text_quality']['valid_records']\n            report['summary']['error_count'] = len(self.validation_results['text_quality']['extraction_errors'])\n        \n        # Assign quality grade\n        if self.quality_score >= 0.9:\n            report['summary']['quality_grade'] = 'A'\n        elif self.quality_score >= 0.8:\n            report['summary']['quality_grade'] = 'B'\n        elif self.quality_score >= 0.7:\n            report['summary']['quality_grade'] = 'C'\n        elif self.quality_score >= 0.6:\n            report['summary']['quality_grade'] = 'D'\n        else:\n            report['summary']['quality_grade'] = 'F'\n        \n        # Save detailed report\n        report_path = output_dir / 'data_quality_report.json'\n        with open(report_path, 'w') as f:\n            json.dump(report, f, indent=2, default=str)\n        \n        # Create human-readable summary\n        summary_path = output_dir / 'quality_summary.txt'\n        with open(summary_path, 'w') as f:\n            f.write(\"=\" * 60 + \"\\n\")\n            f.write(\"DATA QUALITY VALIDATION REPORT\\n\")\n            f.write(\"=\" * 60 + \"\\n\\n\")\n            f.write(f\"Generated: {report['timestamp']}\\n\")\n            f.write(f\"Overall Quality Score: {self.quality_score:.2f}\\n\")\n            f.write(f\"Quality Grade: {report['summary']['quality_grade']}\\n\\n\")\n            \n            f.write(\"SUMMARY:\\n\")\n            f.write(\"-\" * 30 + \"\\n\")\n            f.write(f\"Files Analyzed: {report['summary']['total_files_analyzed']}\\n\")\n            f.write(f\"Valid Records: {report['summary']['valid_records']}\\n\")\n            f.write(f\"Errors Found: {report['summary']['error_count']}\\n\\n\")\n            \n            if self.recommendations:\n                f.write(\"RECOMMENDATIONS:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                for i, rec in enumerate(self.recommendations, 1):\n                    f.write(f\"{i}. {rec}\\n\")\n        \n        logger.info(f\"Quality report saved: {report_path}\")\n        logger.info(f\"Quality summary saved: {summary_path}\")\n        \n        return report

def main():\n    \"\"\"Main function for data quality validation\"\"\"\n    parser = argparse.ArgumentParser(\n        description=\"Data Quality Validator for Resume Analysis\"\n    )\n    \n    parser.add_argument(\"--data-file\", required=True,\n                       help=\"CSV file with resume data to validate\")\n    parser.add_argument(\"--output-dir\", default=\"quality_validation\",\n                       help=\"Output directory for validation results\")\n    parser.add_argument(\"--create-plots\", action=\"store_true\",\n                       help=\"Create visualization plots\")\n    \n    args = parser.parse_args()\n    \n    # Setup output directory\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)\n    \n    logger.info(\"=\" * 50)\n    logger.info(\"DATA QUALITY VALIDATION\")\n    logger.info(\"=\" * 50)\n    \n    try:\n        # Load data\n        logger.info(f\"Loading data from: {args.data_file}\")\n        df = pd.read_csv(args.data_file)\n        logger.info(f\"Loaded {len(df)} records\")\n        \n        # Initialize validator\n        validator = DataQualityValidator()\n        \n        # Extract file paths if available\n        file_paths = []\n        if 'filename' in df.columns:\n            file_paths = [Path(fp) for fp in df['filename'] if pd.notna(fp)]\n        \n        # Run validations\n        if file_paths:\n            validator.validation_results['file_integrity'] = validator.validate_file_integrity(file_paths)\n        \n        validator.validation_results['text_quality'] = validator.validate_text_extraction_quality(df)\n        validator.validation_results['completeness'] = validator.validate_data_completeness(df)\n        validator.validation_results['outliers'] = validator.detect_outliers_and_anomalies(df)\n        \n        # Calculate overall quality score\n        quality_score = validator.calculate_overall_quality_score()\n        logger.info(f\"Overall Quality Score: {quality_score:.2f}\")\n        \n        # Generate recommendations\n        recommendations = validator.generate_recommendations()\n        logger.info(f\"Generated {len(recommendations)} recommendations\")\n        \n        # Create visualizations if requested\n        if args.create_plots:\n            validator.create_visualizations(df, output_dir)\n        \n        # Generate final report\n        report = validator.generate_report(output_dir)\n        \n        logger.info(\"=\" * 50)\n        logger.info(\"VALIDATION COMPLETE!\")\n        logger.info(f\"Quality Grade: {report['summary']['quality_grade']}\")\n        logger.info(f\"Results saved in: {output_dir}\")\n        logger.info(\"=\" * 50)\n        \n    except Exception as e:\n        logger.error(f\"Validation failed: {e}\")\n        raise

if __name__ == \"__main__\":\n    main()
