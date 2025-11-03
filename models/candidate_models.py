from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from .base_model import TinyMLModel
from data import get_multitask_dataloaders  # 导入数据加载器

# 设置随机数种子
SEED = 42  # 你可以选择任何整数作为种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

@dataclass
class CandidateModel:
    """
    表示一个候选神经网络架构及其评估指标
    
    属性:
        config: 模型架构配置字典
        candidate_id: 候选模型唯一标识符 (字符串)
        accuracy: 验证准确率 (0-1)
        latency: 推理延迟 (ms)
        generation: 进化算法中的生成代数
        parent_ids: 父代ID列表 (用于遗传算法)
        metadata: 其他元数据
    """
    config: Dict[str, Any]
    candidate_id: Optional[str] = None  # 新增
    accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None  # 修改为单一的验证准确率
    generation: Optional[int] = None
    parent_ids: Optional[List[int]] = None
    metrics: Optional[Dict[str, Any]] = None
    latency: Optional[float] = None  # 新增推理时延字段 （单位：毫秒）
    peak_memory: Optional[float] = None  # 新增峰值内存字段 （单位：MB）
    estimate_total_size: Optional[float] = None # 自测的未量化内存 （单位：MB)
    comparison_metrics: Optional[Dict[str, float]] = None  # ⭐ 新增：用于Pareto比较的指标
    use_quantized_metrics: Optional[bool] = None  # ⭐ 新增：是否使用量化指标进行比较

    def __post_init__(self):
        """数据验证和默认值设置"""
        self.parent_ids = self.parent_ids or []
        self.metrics = self.metrics or {}
        self.val_accuracy = self.val_accuracy or {}  # 初始化为一个空字典
        self.comparison_metrics = self.comparison_metrics or {}  # ⭐ 新增：初始化比较指标
        self.use_quantized_metrics = self.use_quantized_metrics or False  # ⭐ 新增：默认不使用量化指标

        # 如果没有提供 candidate_id，生成一个默认的
        if self.candidate_id is None:
            import uuid
            self.candidate_id = f"candidate_{uuid.uuid4().hex[:8]}"


    def build_model(self) -> nn.Module:
        """将配置转换为PyTorch模型"""
        model = TinyMLModel(self.config)

        # 确保模型有 output_dim 属性
        if not hasattr(model, 'output_dim'):
            model.output_dim = self._calculate_output_dim()

        return model

    def _calculate_output_dim(self) -> int:
        """根据配置计算最终输出维度"""
        if 'stages' not in self.config:
            return 64  # 默认值
        
        # 取最后一个 stage 的通道数作为输出维度
        last_stage = self.config['stages'][-1]
        return int(last_stage['channels'])
    

    def measure_peak_memory(self, device='cuda', dataset_names=None) -> float:
        """
        测量模型运行时的峰值内存（单位：MB）
        """
        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # 根据实际路径调整
        if dataset_names is None:
            dataset_names = list(dataloaders.keys())  # 默认使用所有数据集
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # 如果是字符串，将其包装为列表
        elif not isinstance(dataset_names, list):
            raise ValueError(f"Invalid dataset_names type: {type(dataset_names)}")

        model = self.build_model().to(device)
        model.eval()

        total_peak_memory = 0
        total_samples = 0
        max_memory = 0

        for dataset_name in dataset_names:
            print(f"测量数据集 {dataset_name} 的峰值内存...")
            dataloader = dataloaders[dataset_name]['train']
            dataset_peak_memory = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # 只测量前 100 条数据
                    break

                inputs = inputs.to(device)

                if device == 'cuda':
                    # 清空显存缓存并重置统计
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取峰值内存
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为 MB
                elif device == 'cpu':
                    import tracemalloc
                    tracemalloc.start()

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取内存使用
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory = peak / (1024 ** 2)  # 转换为 MB
                else:
                    raise ValueError(f"Unsupported device: {device}")

                dataset_peak_memory += peak_memory
                total_samples += 1
                if max_memory < peak_memory:    
                    max_memory = peak_memory

            avg_dataset_peak_memory = dataset_peak_memory / min(100, len(dataloader))
            print(f"数据集 {dataset_name} 的平均峰值内存: {avg_dataset_peak_memory:.2f} MB")
            print(f"数据集 {dataset_name} 的最大峰值内存: {max_memory:.2f} MB")
            total_peak_memory += avg_dataset_peak_memory

        # self.peak_memory = total_peak_memory / len(dataset_names)  # 所有数据集的平均峰值内存
        self.peak_memory = max_memory
        return self.peak_memory
    
    def measure_latency(self, device='cuda', num_runs=10, dataset_names=None) -> float:
        """
        测量模型在指定设备上的推理时延（单位：毫秒）
        
        参数:
            device: 测量设备（'cuda' 或 'cpu'）
            num_runs: 测量次数（取平均值）
            input_shape: 输入张量形状 (batch, channels, time_steps)
            
        返回:
            float: 平均推理时延（毫秒）
        """
        if self.latency is not None:
            return self.latency
        
        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # 根据实际路径调整
        if dataset_names is None:
            dataset_names = dataloaders.keys()  # 默认使用所有数据集
        elif dataset_names and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        elif dataset_names and isinstance(dataset_names, list):
            dataset_names = dataset_names
        model = self.build_model().to(device)
        model.eval()

        total_latency = 0
        total_samples = 0
        
        for dataset_name in dataset_names:
            print(f"测量数据集 {dataset_name} 的推理延迟...")
            dataloader = dataloaders[dataset_name]['train'] 
            dataset_latency = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # 只测量前 100 条数据
                    break

                inputs = inputs.to(device)

                # Warmup（避免冷启动误差）
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(inputs)

                # 正式测量
                start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

                if device == 'cuda':
                    torch.cuda.synchronize()
                    start_time.record()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            _ = model(inputs)
                    end_time.record()
                    torch.cuda.synchronize()
                    latency_ms = start_time.elapsed_time(end_time) / num_runs
                else:
                    import time
                    start = time.time()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            _ = model(inputs)
                    latency_ms = (time.time() - start) * 1000 / num_runs

                dataset_latency += latency_ms
                total_samples += 1

            avg_dataset_latency = dataset_latency / min(100, len(dataloader))
            print(f"数据集 {dataset_name} 的平均推理延迟: {avg_dataset_latency:.2f} ms")
            total_latency += avg_dataset_latency
        self.latency = total_latency / len(dataset_names)  # 所有数据集的平均延迟
        return latency_ms        


    @classmethod
    def load(cls, file_path: str) -> "CandidateModel":
        """从JSON文件加载候选模型"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            config=data["config"],
            candidate_id=data.get("candidate_id"),
            accuracy=data["metrics"].get("accuracy"),
            latency=data["metrics"].get("latency"),
            generation=data.get("generation"),
            parent_ids=data.get("parent_ids", []),
            metrics=data.get("metrics", {})
        )
    def get_details(self) -> Dict[str, Any]:
        """
        返回CandidateModel的详细信息，包括config和所有主要属性。
        """
        return {
            "config": self.config,
            "candidate_id": self.candidate_id,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "peak_memory": self.peak_memory,
            "val_accuracy": self.val_accuracy,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metrics": self.metrics,
        }

    
    
# 测试1D模型的构建
# test_config = {
#     "input_channels": 6,     
#     "num_classes": 12,      
#     "stages": [
#         {
#             "blocks": [
#                 {
#                     "type": "DWSepConv",
#                     "kernel_size": 3,
#                     "stride": 2,
#                     "has_se": False,
#                     "activation": "ReLU6"
#                 }
#             ],
#             "channels": 16
#         },
#         {
#             "blocks": [
#                 {
#                     "type": "MBConv",
#                     "kernel_size": 5,
#                     "expansion": 4,
#                     "stride": 1,
#                     "has_se": True,
#                     "activation": "Swish"
#                 }
#             ],
#             "channels": 32
#         }
#     ]
# }

# model = TinyMLModel(test_config)
# dummy_input = torch.randn(2, 6, 500)  # (B, C, T)
# output = model(dummy_input)
# print(output.shape)  # 预期输出: torch.Size([2, 10])
# test_config = json.loads(test_config)
# model = CandidateModel(test_config)



# 调用方法
# metrics = {
#     'accuracy': candidate.evaluate_accuracy(),  # 新增方法
# }