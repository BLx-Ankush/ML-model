"""
ML Model Efficiency Dashboard
Comprehensive overview of model optimization results and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import os
from datetime import datetime

class EfficiencyDashboard:
    """
    Dashboard for visualizing and comparing model efficiency optimizations
    """
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.models_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load information about all saved models"""
        models_dir = Path(self.output_dir) / "optimized_models"
        
        if not models_dir.exists():
            print("No optimized models found.")
            return
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    self.models_info[model_dir.name] = metadata
    
    def generate_efficiency_report(self):
        """Generate comprehensive efficiency report"""
        print("=" * 80)
        print("🚀 ML MODEL EFFICIENCY DASHBOARD")
        print("=" * 80)
        
        if not self.models_info:
            print("No optimized models found. Run efficiency optimization first!")
            return
        
        # Summary statistics
        total_models = len(self.models_info)
        avg_feature_reduction = np.mean([
            info['results']['feature_reduction_pct'] 
            for info in self.models_info.values()
        ])
        
        excellent_models = sum([
            1 for info in self.models_info.values() 
            if info['results']['efficiency_grade'] == 'EXCELLENT'
        ])
        
        print(f"📊 SUMMARY STATISTICS")
        print(f"   Total Optimized Models: {total_models}")
        print(f"   Average Feature Reduction: {avg_feature_reduction:.1f}%")
        print(f"   Excellent Efficiency Models: {excellent_models}/{total_models}")
        print()
        
        # Individual model reports
        print(f"📈 MODEL EFFICIENCY BREAKDOWN")
        print("-" * 80)
        
        for model_name, info in self.models_info.items():
            results = info['results']
            
            print(f"\n🏷️  Model: {model_name}")
            print(f"   Created: {info['created_at'][:19].replace('T', ' ')}")
            print(f"   Target: {info['target_column']}")
            print(f"   Efficiency Grade: {results['efficiency_grade']} ⭐")
            print(f"   Performance (R²): {results['optimized_r2']:.4f}")
            print(f"   Feature Reduction: {results['feature_reduction_pct']:.1f}% ({results['selected_features']}/{results['original_features']})")
            print(f"   Model Config: {results['model_config']}")
            
            # Performance retention if available
            if 'performance_retention_pct' in results:
                print(f"   Performance Retention: {results['performance_retention_pct']:.1f}%")
        
        print("\n" + "=" * 80)
        return self.models_info
    
    def create_efficiency_visualizations(self):
        """Create efficiency visualization charts"""
        if not self.models_info:
            print("No models to visualize")
            return
        
        # Prepare data for visualization
        models_data = []
        for model_name, info in self.models_info.items():
            results = info['results']
            models_data.append({
                'Model': model_name,
                'R² Score': results['optimized_r2'],
                'Feature Reduction %': results['feature_reduction_pct'],
                'Efficiency Grade': results['efficiency_grade'],
                'Selected Features': results['selected_features'],
                'Original Features': results['original_features'],
                'Performance Retention %': results.get('performance_retention_pct', 100)
            })
        
        df = pd.DataFrame(models_data)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 ML Model Efficiency Dashboard', fontsize=16, fontweight='bold')
        
        # 1. R² Score vs Feature Reduction
        scatter = ax1.scatter(df['Feature Reduction %'], df['R² Score'], 
                            c=df['R² Score'], cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Feature Reduction (%)')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Performance vs Efficiency Trade-off')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1)
        
        # Add model names as annotations
        for i, row in df.iterrows():
            ax1.annotate(row['Model'][:10], (row['Feature Reduction %'], row['R² Score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Efficiency Grades Distribution
        grade_counts = df['Efficiency Grade'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#FF6347', '#708090']  # Green, Gold, Red, Gray
        wedges, texts, autotexts = ax2.pie(grade_counts.values, labels=grade_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(grade_counts)])
        ax2.set_title('Efficiency Grade Distribution')
        
        # 3. Feature Count Comparison
        x_pos = np.arange(len(df))
        width = 0.35
        
        ax3.bar(x_pos - width/2, df['Original Features'], width, 
               label='Original Features', color='lightcoral', alpha=0.7)
        ax3.bar(x_pos + width/2, df['Selected Features'], width, 
               label='Selected Features', color='skyblue', alpha=0.7)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Number of Features')
        ax3.set_title('Feature Count: Before vs After Optimization')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([name[:10] for name in df['Model']], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Retention
        bars = ax4.bar(df['Model'], df['Performance Retention %'], 
                      color=['green' if x >= 95 else 'orange' if x >= 90 else 'red' 
                            for x in df['Performance Retention %']])
        ax4.set_ylabel('Performance Retention (%)')
        ax4.set_title('Performance Retention After Optimization')
        ax4.set_xticklabels([name[:10] for name in df['Model']], rotation=45)
        ax4.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Excellent (95%+)')
        ax4.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Good (90%+)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add values on top of bars
        for bar, value in zip(bars, df['Performance Retention %']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(self.output_dir) / "efficiency_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Efficiency dashboard saved to: {output_path}")
        
        plt.show()
        
        return df
    
    def compare_models(self, model_names=None):
        """Compare specific models in detail"""
        if model_names is None:
            model_names = list(self.models_info.keys())
        
        print("🔍 DETAILED MODEL COMPARISON")
        print("-" * 60)
        
        comparison_data = []
        for model_name in model_names:
            if model_name in self.models_info:
                info = self.models_info[model_name]
                results = info['results']
                
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': f"{results['optimized_r2']:.4f}",
                    'RMSE': f"{results['optimized_rmse']:.3f}",
                    'Features': f"{results['selected_features']}/{results['original_features']}",
                    'Reduction %': f"{results['feature_reduction_pct']:.1f}%",
                    'Grade': results['efficiency_grade'],
                    'Config': results['model_config']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_optimization_recommendations(self):
        """Get recommendations for further optimization"""
        print("\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 60)
        
        recommendations = []
        
        for model_name, info in self.models_info.items():
            results = info['results']
            model_recommendations = []
            
            # Performance recommendations
            if results['optimized_r2'] < 0.90:
                model_recommendations.append("⚠️  Consider more features or different algorithms")
            
            # Efficiency recommendations
            if results['feature_reduction_pct'] < 30:
                model_recommendations.append("🔧 Try more aggressive feature selection")
            
            if results['efficiency_grade'] == 'MINIMAL':
                model_recommendations.append("⚡ Experiment with different selection methods")
            
            # Configuration recommendations
            if results['model_config'] == 'accurate':
                model_recommendations.append("🚀 Try 'balanced' config for better speed")
            
            if model_recommendations:
                print(f"\n📋 {model_name}:")
                for rec in model_recommendations:
                    print(f"   {rec}")
            else:
                print(f"\n✅ {model_name}: Well optimized!")
        
        # General recommendations
        print(f"\n🎯 GENERAL RECOMMENDATIONS:")
        
        avg_reduction = np.mean([
            info['results']['feature_reduction_pct'] 
            for info in self.models_info.values()
        ])
        
        if avg_reduction < 25:
            print("   • Consider ensemble feature selection methods")
            print("   • Try correlation-based feature removal")
        
        excellent_count = sum([
            1 for info in self.models_info.values() 
            if info['results']['efficiency_grade'] == 'EXCELLENT'
        ])
        
        if excellent_count == len(self.models_info):
            print("   • 🎉 All models are excellently optimized!")
            print("   • Consider A/B testing in production")
        
        return recommendations

def generate_complete_report():
    """Generate complete efficiency report and dashboard"""
    dashboard = EfficiencyDashboard()
    
    # Generate text report
    models_info = dashboard.generate_efficiency_report()
    
    # Create visualizations
    if models_info:
        df = dashboard.create_efficiency_visualizations()
        
        # Model comparison
        print("\n")
        comparison_df = dashboard.compare_models()
        
        # Recommendations
        dashboard.get_optimization_recommendations()
        
        return dashboard, df, comparison_df
    
    return dashboard, None, None

if __name__ == "__main__":
    # Generate complete dashboard
    dashboard, viz_df, comp_df = generate_complete_report()
    
    print(f"\n🎉 Dashboard generation complete!")
    print(f"Check the 'outputs/efficiency_dashboard.png' for visual insights.")