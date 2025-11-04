"""
配置降级策略模块
用于在内存超标时自动降级模型配置
"""
import json
from typing import Dict, Any, Optional, Callable, Tuple
from models import CandidateModel
from utils import calculate_memory_usage

class ConfigDegradationManager:
    """配置降级管理器"""
    
    def __init__(self, dataset_info: Dict[str, Any], max_memory: float,
                 check_memory_fn: Callable, is_duplicate_fn: Callable):
        """
        初始化降级管理器
        
        Args:
            dataset_info: 数据集信息
            max_memory: 最大内存限制(MB)
            check_memory_fn: 内存检查函数
            is_duplicate_fn: 重复检查函数
        """
        self.dataset_info = dataset_info
        self.max_memory = max_memory
        self.check_memory_fn = check_memory_fn
        self.is_duplicate_fn = is_duplicate_fn
        
        # 降级策略列表
        self.strategies = [
            self.apply_stage_reduction,
            self.apply_conv_type_optimization,
            self.apply_channel_reduction,
            self.apply_block_reduction
        ]

    def generate_degraded_config(self, base_config: Dict[str, Any], 
                                 direction: str, current_memory: float) -> Dict[str, Any]:
        """
        生成降级配置
        
        Args:
            base_config: 基础配置
            direction: 量化方向
            current_memory: 当前内存使用(MB)
            
        Returns:
            降级后的配置
        """
        print("🛠️ 使用智能降级生成安全配置")
        
        # 创建安全配置
        safe_config = base_config.copy() if base_config else {}
        safe_config["quant_mode"] = direction
        
        # 如果没有基础配置，生成最小配置
        if not safe_config.get("stages"):
            safe_config = self._create_minimal_config(direction)
            if not self.is_duplicate_fn(safe_config):
                return safe_config
        
        # 依次尝试降级策略
        for strategy in self.strategies:
            temp_config = strategy(safe_config.copy(), current_memory)
            if temp_config and not self.is_duplicate_fn(temp_config):
                memory_ok, memory_usage, _ = self.check_memory_fn(temp_config)
                if memory_ok:
                    print(f"✅ 降级策略成功: 内存={memory_usage:.2f}MB")
                    return temp_config
        
        # 最终手段：绝对最小配置
        return self._create_minimal_config_with_retry(direction)
    
    def _create_minimal_config(self, direction: str) -> Dict[str, Any]:
        """创建最小配置"""
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "SeDpConv",  # 最小内存的卷积类型
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "activation": "ReLU",
                            "skip_connection": True
                        }
                    ],
                    "channels": 8
                }
            ]
        }
    
    def _create_minimal_config_with_retry(self, direction: str) -> Dict[str, Any]:
        """创建最小配置并处理重复"""
        print("🚨 使用绝对最小配置")
        minimal_config = self._create_minimal_config(direction)
        
        # 如果重复，稍微修改
        attempt = 0
        while self.is_duplicate_fn(minimal_config) and attempt < 10:
            minimal_config["stages"][0]["channels"] += 1
            attempt += 1
        
        return minimal_config
    
    # ========== 降级策略 ==========
    
    def apply_stage_reduction(self, config: Dict[str, Any], 
                              current_memory: float) -> Optional[Dict[str, Any]]:
        """策略1: 减少stage数量"""
        if current_memory <= self.max_memory:
            return None
        
        original_stage_count = len(config.get("stages", []))
        if original_stage_count <= 1:
            return None
        
        print("🔧 策略1: 减少stage数量")
        temp_config = config.copy()
        
        for stage_count in range(original_stage_count - 1, 0, -1):
            temp_config["stages"] = temp_config["stages"][:stage_count]
            memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
            
            if memory_ok:
                print(f"✅ Stage: {original_stage_count} → {stage_count}, 内存: {new_memory:.2f}MB")
                return temp_config
        
        return None
    
    def apply_conv_type_optimization(self, config: Dict[str, Any], 
                                     current_memory: float) -> Optional[Dict[str, Any]]:
        """策略2: 优化卷积类型"""
        if current_memory <= self.max_memory:
            return None
        
        print("🔧 策略2: 优化卷积类型")
        conv_type_priority = ["MBConv", "DWSepConv", "SeSepConv", "DpConv", "SeDpConv"]
        
        temp_config = json.loads(json.dumps(config))  # 深拷贝
        
        # 从后往前处理stage
        for stage_idx in range(len(temp_config.get("stages", [])) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            
            for block_idx in range(len(stage.get("blocks", [])) - 1, -1, -1):
                block = stage["blocks"][block_idx]
                current_type = block.get("type", "MBConv")
                
                if current_type == "SeDpConv":
                    continue
                
                try:
                    current_priority = conv_type_priority.index(current_type)
                except ValueError:
                    current_priority = 0
                
                # 尝试更小的类型
                for smaller_type in conv_type_priority[current_priority + 1:]:
                    block["type"] = smaller_type
                    memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                    
                    if memory_ok:
                        print(f"✅ Stage{stage_idx}-Block{block_idx}: {current_type} → {smaller_type}")
                        return temp_config
        
        return None
    
    def apply_channel_reduction(self, config: Dict[str, Any], 
                                current_memory: float) -> Optional[Dict[str, Any]]:
        """策略3: 减少通道数"""
        if current_memory <= self.max_memory:
            return None
        
        print("🔧 策略3: 减少通道数")
        
        for stage_idx in range(len(config.get("stages", [])) - 1, -1, -1):
            stage = config["stages"][stage_idx]
            original_channels = stage["channels"]
            
            for reduction_factor in [0.75, 0.5, 0.25]:
                new_channels = max(8, int(original_channels * reduction_factor))
                if new_channels == stage["channels"]:
                    continue
                
                stage["channels"] = new_channels
                memory_ok, new_memory, _ = self.check_memory_fn(config)
                
                if memory_ok:
                    print(f"✅ Stage{stage_idx}通道: {original_channels} → {new_channels}")
                    return config
            
            stage["channels"] = original_channels
        
        return None
    
    def apply_block_reduction(self, config: Dict[str, Any], 
                              current_memory: float) -> Optional[Dict[str, Any]]:
        """策略4: 减少block数量"""
        if current_memory <= self.max_memory:
            return None
        
        print("🔧 策略4: 减少block数量")
        temp_config = json.loads(json.dumps(config))
        
        for stage_idx in range(len(temp_config["stages"]) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            original_block_count = len(stage["blocks"])
            
            if original_block_count <= 1:
                if stage_idx > 0:
                    print(f"🔄 Stage{stage_idx} 只有1个block，删除整个stage")
                    temp_config["stages"].pop(stage_idx)
                    memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                    if memory_ok:
                        print(f"✅ 删除Stage{stage_idx}后内存达标: {new_memory:.2f}MB")
                        return temp_config
                continue
            
            for new_block_count in range(original_block_count - 1, 0, -1):
                stage["blocks"] = stage["blocks"][:new_block_count]
                memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                
                if memory_ok:
                    print(f"✅ Stage{stage_idx} block: {original_block_count} → {new_block_count}")
                    return temp_config
        
        return None
