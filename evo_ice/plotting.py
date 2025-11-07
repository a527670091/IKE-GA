import matplotlib.pyplot as plt
import os
import numpy as np

def plot_performance_curves(history, output_dir):
    """
    根据记录的进化历史数据绘制性能曲线图。

    参数:
    - history (dict): 包含每一代性能指标的字典。
    - output_dir (str): 保存图像的目标目录。
    """
    generations = history['generations']
    eval_counts = history['eval_counts']
    avg_efficacy = history['avg_efficacy']
    avg_generalization = history['avg_generalization']
    avg_specificity = history['avg_specificity']
    best_efficacy = history['best_efficacy']
    best_generalization = history['best_generalization']
    best_specificity = history['best_specificity']

    # 绘制性能 vs. 进化代数
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Performance over Generations', fontsize=16)

    # 绘制平均性能
    axes[0].plot(generations, avg_efficacy, marker='o', linestyle='-', label='Avg Efficacy')
    axes[0].plot(generations, avg_generalization, marker='s', linestyle='-', label='Avg Generalization')
    axes[0].plot(generations, avg_specificity, marker='^', linestyle='-', label='Avg Specificity')
    axes[0].set_title('Average Fitness Scores per Generation')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Average Score')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(generations)


    # 绘制最佳性能
    axes[1].plot(generations, best_efficacy, marker='o', linestyle='-', label='Best Efficacy')
    axes[1].plot(generations, best_generalization, marker='s', linestyle='-', label='Best Generalization')
    axes[1].plot(generations, best_specificity, marker='^', linestyle='-', label='Best Specificity')
    axes[1].set_title('Best Fitness Scores per Generation')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Best Score')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(generations)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    gen_plot_path = os.path.join(output_dir, 'performance_over_generations.png')
    plt.savefig(gen_plot_path)
    plt.close()
    print(f"Saved generation performance plot to {gen_plot_path}")

    # 绘制性能 vs. 评估次数
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Performance over Evaluations', fontsize=16)

    # 绘制平均性能
    axes[0].plot(eval_counts, avg_efficacy, marker='o', linestyle='-', label='Avg Efficacy')
    axes[0].plot(eval_counts, avg_generalization, marker='s', linestyle='-', label='Avg Generalization')
    axes[0].plot(eval_counts, avg_specificity, marker='^', linestyle='-', label='Avg Specificity')
    axes[0].set_title('Average Fitness Scores per Evaluation Count')
    axes[0].set_xlabel('Number of Evaluations')
    axes[0].set_ylabel('Average Score')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制最佳性能
    axes[1].plot(eval_counts, best_efficacy, marker='o', linestyle='-', label='Best Efficacy')
    axes[1].plot(eval_counts, best_generalization, marker='s', linestyle='-', label='Best Generalization')
    axes[1].plot(eval_counts, best_specificity, marker='^', linestyle='-', label='Best Specificity')
    axes[1].set_title('Best Fitness Scores per Evaluation Count')
    axes[1].set_xlabel('Number of Evaluations')
    axes[1].set_ylabel('Best Score')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    eval_plot_path = os.path.join(output_dir, 'performance_over_evaluations.png')
    plt.savefig(eval_plot_path)
    plt.close()
    print(f"Saved evaluation performance plot to {eval_plot_path}")
