from typing import List, Dict, Any, Optional, Tuple
from data import get_dataset_info, get_multitask_dataloaders
from queue import Empty
import time
import torch.multiprocessing as mp
from models import CandidateModel
from models import QuantizableModel
from utils import calculate_memory_usage
import copy
import torch
from models import get_quantization_option, apply_configurable_static_quantization
from models import fuse_QATmodel_modules, prepare_qaft_model, apply_qaft_quantization  # ✅ 新增

class ParallelModelEvaluator:
    """
    并行模型评估器 - 工作进程使用
    每个工作进程有一个独立的评估器实例
    """
    def __init__(self, gpu_id: int, constraints: Dict[str, float], 
                 dataset_name: str, train_epochs: int = 100):
        self.gpu_id = gpu_id
        self.constraints = constraints
        self.dataset_name = dataset_name
        self.train_epochs = train_epochs
        
        # 设置设备
        self.device = f"cuda:{gpu_id}"
        
        # 加载数据集
        self.dataset_info = get_dataset_info(dataset_name)
        multitask_dataloaders = get_multitask_dataloaders(
            root_dir="/root/har_train/data/UniMTS_data", 
            datasets=[dataset_name]
        )
        self.dataloader = multitask_dataloaders[dataset_name]
        
        print(f"[GPU {gpu_id}] 评估器初始化完成")
    
    def evaluate(self, config: Dict[str, Any], candidate_id: str) -> Tuple[float, Dict[str, Any]]:
        """评估单个配置"""
        try:
            print(f"[GPU {self.gpu_id}] 开始评估 {candidate_id}")
            
            # 创建候选模型
            candidate = CandidateModel(config=config)
            candidate.candidate_id = candidate_id
            
            # 2. 获取量化模式
            quant_mode = config.get('quant_mode', 'none')
            self._current_quant_mode = quant_mode  # ✅ 保存当前量化模式
            print(f"[GPU {self.gpu_id}] 量化模式: {quant_mode}")

            # 3. 构建模型
            model = candidate.build_model()
            
            # ========== 公平的训练策略 ==========
            # QAFT 准备
            if quant_mode == 'qaft':
                print(f"[GPU {self.gpu_id}] ⚖️ QAFT 模式 - 两阶段训练 (总计{self.train_epochs}轮)")
                
                # 计算轮数分配 (90% + 10%)
                pretrain_epochs = int(self.train_epochs * 0.9)  # 90轮
                finetune_epochs = self.train_epochs - pretrain_epochs  # 10轮
                
                print(f"[GPU {self.gpu_id}] 📊 训练分配:")
                print(f"  - 阶段1(预训练): {pretrain_epochs}轮")
                print(f"  - 阶段2(微调): {finetune_epochs}轮")
                print(f"  - 总计: {pretrain_epochs + finetune_epochs}轮")
                
                # ✅ 阶段1: 正常训练预热
                print(f"[GPU {self.gpu_id}] 🔥 阶段1: 预训练开始...")
                from training import SingleTaskTrainer
                
                pretrain_trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                pretrain_accuracy, pretrain_metrics, _, pretrain_state = pretrain_trainer.train(
                    epochs=pretrain_epochs,
                    save_path=f"checkpoints/pretrain_{candidate_id}.pth"
                )
                
                print(f"[GPU {self.gpu_id}] ✅ 阶段1完成: 准确率={pretrain_accuracy:.2f}%")
                
                # ✅ 阶段2: QAFT 微调
                print(f"[GPU {self.gpu_id}] 🎯 阶段2: QAFT微调开始...")
                
                # 加载最佳预训练权重
                if pretrain_state and 'model' in pretrain_state:
                    model.load_state_dict(pretrain_state['model'])
                    print(f"[GPU {self.gpu_id}] 📥 已加载预训练权重")
                
                # 准备QAFT (插入FakeQuantize)
                model = self._prepare_qat_model(model)
                
                # 微调训练
                qaft_trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = qaft_trainer.train(
                    epochs=finetune_epochs,
                    save_path=f"checkpoints/candidate_{candidate_id}.pth"
                )
                
                print(f"[GPU {self.gpu_id}] ✅ 阶段2完成: 准确率={best_accuracy:.2f}%")
                print(f"[GPU {self.gpu_id}] 📈 准确率变化: {pretrain_accuracy:.2f}% → {best_accuracy:.2f}%")

            elif quant_mode == 'qat':
                # ✅ QAT: 插入量化后完整训练
                print(f"[GPU {self.gpu_id}] ⚡ QAT模式 - 量化感知训练 ({self.train_epochs}轮)")
                model = self._prepare_qat_model(model)
                
                from training import SingleTaskTrainer
                trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = trainer.train(
                    epochs=self.train_epochs,
                    save_path=f"checkpoints/candidate_{candidate_id}.pth"
                )
                
                print(f"[GPU {self.gpu_id}] ✅ QAT 训练完成: 准确率={best_accuracy:.2f}%")

            else:
                # ✅ None/Static: 正常训练
                print(f"[GPU {self.gpu_id}] 🔄 {quant_mode.upper()}模式 - 正常训练 ({self.train_epochs}轮)")
                from training import SingleTaskTrainer
                trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = trainer.train(
                    epochs=self.train_epochs,
                    save_path=f"checkpoints/candidate_{candidate_id}.pth"
                )
                
                print(f"[GPU {self.gpu_id}] ✅ 训练完成: 准确率={best_accuracy:.2f}%")
            
            # ========== 后续代码不变 ==========
            # 6. 测量原始模型性能
            memory_usage = calculate_memory_usage(
                model,
                input_size=(64, self.dataset_info['channels'], self.dataset_info['time_steps']),
                device='cpu'
            )
            print(f"[GPU {self.gpu_id}] 原始内存: {memory_usage['total_memory_MB']:.2f}MB")
            # 奖励就是准确率
            reward = best_accuracy
            accuracy = best_accuracy
            
            metrics = {
                "original_accuracy": accuracy,
                "original_accuracy_percent": best_accuracy,
                "original_memory": memory_usage['total_memory_MB'],
                "reward": reward,
                "train_loss": best_val_metrics.get('loss', 0.0) if best_val_metrics else 0.0,
                # "training_history": history,
                "gpu_id": self.gpu_id,
                'quantization_mode': quant_mode
            }
            
            # 8. 量化处理 (如果需要)
            if quant_mode != 'none':
                quant_accuracy, quant_metrics = self._apply_quantization(
                    model, best_model_state, quant_mode, candidate_id
                )

                accuracy_drop = best_accuracy - quant_accuracy

                # 更新指标
                metrics.update({
                    'quantized_accuracy': quant_metrics.get('accuracy', 0),
                    'quantized_memory': quant_metrics.get('peak_memory', memory_usage['total_memory_MB']),
                    'quantization_method': quant_metrics.get('method', 'unknown'),
                    'accuracy_drop': accuracy_drop,
                    'quantization_save_path': quant_metrics.get('save_path', None)
                })
                
                # 量化后自然是选择量化后的准确率
                reward = quant_accuracy
                metrics['accuracy'] = quant_accuracy
                print(f"[GPU {self.gpu_id}] 使用量化奖励: {reward:.2f}")
                print(f"  - 原始准确率: {best_accuracy:.2f}%")
                print(f"  - 量化准确率: {quant_accuracy:.2f}%")
                print(f"  - 准确率下降: {accuracy_drop:.2f}%")
                print(f"  - 使用量化奖励: {reward:.2f}")
            else:
                reward = best_accuracy
                metrics['accuracy'] = best_accuracy
                print(f"[GPU {self.gpu_id}] 未进行量化，使用原始奖励: {reward:.2f}")
            
            metrics['reward'] = reward
            print(f"[GPU {self.gpu_id}] 完成评估 {candidate_id}: 奖励={reward:.2f}")
            
            return reward, metrics
            
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 评估失败 {candidate_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {
                "error": str(e),
                "accuracy": 0.0,
                "reward": 0.0,
                "gpu_id": self.gpu_id
            }
        
    def _prepare_qat_model(self, model):
        """准备QAT量化感知训练模型"""
        try:

            # 获取量化模式
            quant_mode = getattr(self, '_current_quant_mode', 'qat')
            if quant_mode == 'qaft':
                print(f"[GPU {self.gpu_id}] 准备QAFT量化 (微调模式)")
                model = prepare_qaft_model(model, freeze_backbone=True)
            else:
                print(f"[GPU {self.gpu_id}] 准备QAT量化 (完整训练)")
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                fuse_QATmodel_modules(model)
                model.train()
                torch.quantization.prepare_qat(model, inplace=True)
            print(f"[GPU {self.gpu_id}] 量化准备完成")
            return model
            
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 量化准备失败: {e}")
            import traceback
            traceback.print_exc()
            return model
    
    def _apply_quantization(self, model, best_state: dict, quant_mode: str, 
                           candidate_id: str) -> Tuple[float, Dict[str, Any]]:
        """应用量化并评估"""
        try:
            print(f"[GPU {self.gpu_id}] 应用{quant_mode}量化")
            
            # 根据量化模式选择配置
            if quant_mode == 'static':
                quantization_options = [
                    ('int8_default', '默认INT8量化'),
                    ('int8_per_channel', '逐通道INT8量化'),
                    ('int8_reduce_range', '减少范围INT8量化'),
                    ('int8_asymmetric', 'INT8非对称量化'),
                    ('int8_histogram', 'INT8直方图校准'),
                    ('int8_moving_avg', 'INT8移动平均校准')
                ]
            elif quant_mode == 'qat':
                quantization_options = [('qat_default', 'QAT量化')]
            elif quant_mode == 'qaft':
                # ✅ QAFT使用特殊处理
                print(f"[GPU {self.gpu_id}] QAFT模式 - 直接转换")
                quantization_options = [('qaft_default', 'QAFT量化')]
            else:
                return 0.0, {}
            
            best_accuracy = 0.0
            best_quantized_model = None
            best_memory = 0.0
            best_option_name = ""
            
            # 尝试不同量化配置
            for option_name, option_desc in quantization_options:
                try:
                    print(f"[GPU {self.gpu_id}] 尝试 {option_desc}")
                    
                    quantized_model = self._apply_quantization_helper(
                        model, quant_mode, option_name
                    )
                    
                    if quantized_model:
                        # 创建任务头并加载权重
                        import torch.nn as nn
                        task_head = nn.Linear(
                            model.output_dim,
                            len(self.dataloader['test'].dataset.classes)
                        ).to('cpu')
                        
                        if best_state and 'head' in best_state:
                            task_head.load_state_dict(best_state['head'])
                        
                        # 评估量化模型
                        from models import evaluate_quantized_model
                        quant_accuracy = evaluate_quantized_model(
                            quantized_model, self.dataloader, task_head,
                            f"量化模型({option_name})"
                        )

                         # 测量内存 (只测量一次)
                        quant_memory = calculate_memory_usage(
                            quantized_model,
                            input_size=(64, self.dataset_info['channels'], self.dataset_info['time_steps']),
                            device='cpu'
                        )['total_memory_MB']
                        
                        print(f"[GPU {self.gpu_id}] 📊 {option_desc}: "
                              f"{quant_accuracy:.1f}% / {quant_memory:.2f}MB")
                        
                        # 记录最佳结果
                        if quant_accuracy > best_accuracy:
                            best_accuracy = quant_accuracy
                            best_quantized_model = quantized_model
                            best_memory = quant_memory
                            best_option_name = option_name
                
                except Exception as e:
                    print(f"[GPU {self.gpu_id}] {option_desc} 失败: {e}")
                    continue
            
            # 保存最佳量化模型
            if best_quantized_model:
                import torch
                quant_save_path = f"checkpoints/quant_{candidate_id}_{best_option_name}.pth"
                torch.save(best_quantized_model.state_dict(), quant_save_path)
                
                print(f"[GPU {self.gpu_id}] 最佳量化: {best_option_name}, "
                      f"准确率={best_accuracy:.1f}%")
                
                return best_accuracy, {
                    'accuracy': best_accuracy,
                    'method': best_option_name,
                    'peak_memory': best_memory,
                    'save_path': quant_save_path
                }
            
            return 0.0, {}
            
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 量化失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {}
    
    def _apply_quantization_helper(self, model, quant_mode: str, 
                                   quantization_option: str = 'int8_per_channel'):
        """量化辅助方法"""
        
        model_copy = copy.deepcopy(model)
        model_copy.to('cpu').eval()
        
        if quant_mode == 'static':
            # 静态量化
            quant_config = get_quantization_option(quantization_option)
            print(f"[GPU {self.gpu_id}] 量化配置: {quant_config['description']}")
            
            quantized_model = apply_configurable_static_quantization(
                model_copy,
                self.dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
            
        elif quant_mode in ['qat', 'qaft']:
            # ✅ QAT和QAFT都使用convert转换
            print(f"[GPU {self.gpu_id}] 转换{quant_mode.upper()}模型")
            model_copy.eval()
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
        else:
            print(f"[GPU {self.gpu_id}] 未知量化模式: {quant_mode}")
            return model_copy
            
        return quantized_model


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                   constraints: Dict[str, float], dataset_name: str, 
                   train_epochs: int):
    """
    工作进程函数
    
    参数:
        gpu_id: GPU ID
        task_queue: 任务队列（接收配置）
        result_queue: 结果队列（返回评估结果）
        constraints: 约束条件
        dataset_name: 数据集名称
        train_epochs: 训练轮数
    """
    print(f"[Worker-GPU{gpu_id}] 启动")
    
    # 创建评估器
    evaluator = ParallelModelEvaluator(
        gpu_id=gpu_id,
        constraints=constraints,
        dataset_name=dataset_name,
        train_epochs=train_epochs
    )
    
    while True:
        try:
            # 从队列获取任务（超时60秒）
            task = task_queue.get(timeout=60)
            
            # 检查是否是终止信号
            if task is None:
                print(f"[Worker-GPU{gpu_id}] 收到终止信号")
                break
            
            # 解包任务
            candidate_id, config = task
            
            # 评估
            reward, metrics = evaluator.evaluate(config, candidate_id)
            
            # 返回结果
            result_queue.put((candidate_id, reward, metrics))
            
        except Empty:
            # 队列超时，继续等待
            continue
        except Exception as e:
            print(f"[Worker-GPU{gpu_id}] 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[Worker-GPU{gpu_id}] 退出")