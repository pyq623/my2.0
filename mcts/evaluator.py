# models/evaluator.py
from typing import Dict, Any, Tuple
import numpy as np
from models import CandidateModel
from training import SingleTaskTrainer
import os

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, constraints: Dict[str, float], device: str = "cuda",
                 dataloader: Dict = None, train_epochs: int = 100, save_dir: str = "checkpoints"):
        """
        初始化评估器
        
        参数:
            dataloaders: 包含'train'和'test'的数据加载器字典
            device: 训练设备
            train_epochs: 训练周期数
            save_dir: 模型保存目录
        """
        self.constraints = constraints
        self.device = device
        self.train_epochs = train_epochs
        self.save_dir = save_dir
        self.dataloader = dataloader

        # 检查数据加载器
        if self.dataloader is None:
            print("⚠️ 警告: 没有提供数据加载器，评估将失败")
            raise ValueError("必须提供数据加载器以进行模型评估")

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    def evaluate_candidate(self, candidate: CandidateModel, dataset_names: list = None) -> Tuple[float, Dict[str, Any]]:
        """评估候选模型并返回奖励和指标 - 只基于准确率"""
        try:
            # 构建模型
            model = candidate.build_model()

            # 创建训练器
            trainer = SingleTaskTrainer(
                model=model,
                dataloaders=self.dataloader,
                device=self.device
            )

            # 设置模型保存路径
            model_save_path = os.path.join(
                self.save_dir, 
                f"candidate_{candidate.candidate_id}.pth"
            )

           # 进行训练并获取最佳准确率
            best_accuracy, best_val_metrics, history, best_model_state = trainer.train(
                epochs=self.train_epochs,
                save_path=model_save_path
            )
            
            # 将准确率从百分比转换为0-1范围
            accuracy = best_accuracy / 100.0
            
            # 奖励就是准确率（0-100范围）
            reward = best_accuracy  # 已经是百分比形式
            
            metrics = {
                "accuracy": accuracy,
                "accuracy_percent": best_accuracy,  # 百分比形式
                "reward": reward,
                "train_loss": best_val_metrics.get('loss', 0.0) if best_val_metrics else 0.0,
                "training_history": history,  # 包含完整的训练历史
                "model_save_path": model_save_path,
                "training_epochs": self.train_epochs
            }
            
            # 更新候选模型指标
            candidate.accuracy = accuracy
            candidate.metrics = metrics

            # 保存最佳模型状态到候选对象（可选）
            if best_model_state:
                candidate.best_model_state = best_model_state
            
            print(f"✅ 候选模型评估完成: 准确率 = {best_accuracy:.2f}%, 奖励 = {reward:.2f}")
            
            return reward, metrics
            
        except Exception as e:
            print(f"评估模型时出错: {e}")
            # 返回惩罚奖励
            return 0.0, {"error": str(e), "accuracy": 0.0, "accuracy_percent": 0.0, "reward": 0.0, "train_loss": 0.0}
