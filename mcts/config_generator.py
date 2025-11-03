# models/config_generator.py
import json
import random
from typing import Dict, Any, List
from models import CandidateModel
from utils import calculate_memory_usage

class ConfigGenerator:
    """模型配置生成器"""
    
    def __init__(self, search_space: Dict[str, Any], dataset_info: Dict[str, Any], max_memory: float = 20.0):
        self.search_space = search_space
        self.dataset_info = dataset_info
        self.max_memory = max_memory  # MB
        
    def generate_random_config(self, quant_mode: str = "none") -> Dict[str, Any]:
        """生成随机模型配置，确保满足内存约束"""
        max_attempts = 10
        for attempt in range(max_attempts):
            config = self._try_generate_random_config(quant_mode)
            if self._check_memory_constraint(config):
                return config
        
        # 如果多次尝试都失败，返回最小配置
        return self._generate_minimal_config(quant_mode)
    
    def _try_generate_random_config(self, quant_mode: str = "none") -> Dict[str, Any]:
        """生成随机模型配置"""
        # 限制阶段数量和 block 数量来满足内存约束
        max_stages = min(4, max(self.search_space["stages"]))
        max_blocks_per_stage = min(2, max(self.search_space["blocks_per_stage"]))

        num_stages = random.choice([s for s in self.search_space["stages"] if s <= max_stages])
        
        stages = []
        in_channels = self.dataset_info.get("channels", 6)
        classes = self.dataset_info.get("num_classes", 10)
        current_channels = in_channels
        
        for i in range(num_stages):
            stage_channels = random.choice(self.search_space["channels"])
            num_blocks = random.choice([b for b in self.search_space["blocks_per_stage"] if b <= max_blocks_per_stage])
            
            blocks = []
            for j in range(num_blocks):
                is_first_block = (j == 0)
                block = self._generate_block(current_channels, stage_channels, is_first_block)
                blocks.append(block)
                
                # 更新当前通道数
                if block["type"] == "SeDpConv":
                    current_channels = stage_channels  # SeDpConv保持通道数不变
                else:
                    current_channels = stage_channels
            
            stage_config = {
                "blocks": blocks,
                "channels": stage_channels
            }
            stages.append(stage_config)
        
        return {
            "input_channels": in_channels,
            "num_classes": classes,
            "quant_mode": quant_mode,
            "stages": stages
        }
    
    def _generate_block(self, in_channels: int, stage_channels: int, is_first_block: bool) -> Dict[str, Any]:
        """生成单个block配置"""
        block_type = random.choice(self.search_space["conv_types"])
        
        # 处理 SeDpConv 的特殊约束
        if block_type == "SeDpConv":
            # SeDpConv 要求输入输出通道相等
            stage_channels = in_channels
            # 如果是第一个 block ，通道数必须等于 input_channels
            if is_first_block:
                stage_channels = self.dataset_info.get("channels", 6)
        
        # 处理 MBConv 和 DWSepConv 的关系
        if block_type == "MBConv":
            expansion = random.choice([e for e in self.search_space["expansions"] if 1 < e <= 3])
        elif block_type == "DWSepConv":
            expansion = 1
        else:
            expansion = random.choice([e for e in self.search_space["expansions"] if e <= 2])  # 限制expansion
        
        # 处理SE模块
        has_se = random.choice(self.search_space["has_se"])
        if has_se:
            se_ratio = random.choice([r for r in self.search_space["se_ratios"] if r > 0])
        else:
            se_ratio = 0
        
        # 对于 SeDpConv 和 SeSepConv ，强制启用 SE
        if block_type in ["SeDpConv", "SeSepConv"]:
            has_se = True
            se_ratio = random.choice([r for r in self.search_space["se_ratios"] if r > 0])
        
        # 处理 skip connection 规则
        if block_type in ["DpConv", "SeDpConv", "SeSepConv"]:
            # DpConv, SeDpConv和SeSepConv不支持skip connection
            skip_connection = False
        else:
            # DWSepConv、MBConv支持skip connection
            skip_connection = random.choice(self.search_space["skip_connection"])

        return {
            "type": block_type,
            "kernel_size": random.choice(self.search_space["kernel_sizes"]),
            "stride": random.choice(self.search_space["strides"]) if is_first_block else 1,
            "expansion": expansion,
            "has_se": has_se,
            "se_ratio": se_ratio,
            "skip_connection": skip_connection,
            "activation": random.choice(self.search_space["activations"])
        }
    
    def _check_memory_constraint(self, config: Dict[str, Any]) -> bool:
        """检查配置的内存使用情况"""
        try:
            candidate = CandidateModel(config=config)
            model = candidate.build_model()
            memory_info = calculate_memory_usage(model, input_size=(64, self.dataset_info['channels'], self.dataset_info['time_steps']), device='cuda')
            memory_usage = memory_info["total_memory_MB"]
            
            # 根据量化模式调整内存使用量
            quant_mode = config.get("quant_mode", "none")
            if quant_mode in ["static", "qat"]:
                # 量化模型通常可以压缩到原来的 1/4 左右
                compressed_memory = memory_usage / 4.0
                print(f"量化模型内存压缩: {memory_usage:.2f}MB → {compressed_memory:.2f}MB (quant_mode: {quant_mode})")
                memory_usage = compressed_memory

            return memory_usage <= self.max_memory
        except Exception:
            return False
        
    def _generate_minimal_config(self, quant_mode: str) -> Dict[str, Any]:
        """生成最小有效配置（保底方案）"""
        in_channels = self.dataset_info.get("channels", 6)
        classes = self.dataset_info.get("num_classes", 10)
        
        return {
            "input_channels": in_channels,
            "num_classes": classes,
            "quant_mode": quant_mode,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "SeDpConv",
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratio": 0,
                            "skip_connection": False,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": in_channels
                }
            ]
        }
    
    def mutate_config(self, base_config: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """变异现有配置，确保满足内存约束"""
        max_attempts = 10
        for attempt in range(max_attempts):
            mutated_config = self._try_mutate_config(base_config, mutation_rate)
            if self._check_memory_constraint(mutated_config):
                return mutated_config
        
        # 如果变异后不满足约束，返回原始配置
        return base_config
    
    def _try_mutate_config(self, base_config: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """变异现有配置"""
        mutated_config = json.loads(json.dumps(base_config))  # 深拷贝
        
        # 变异阶段数量 (更保守)
        if random.random() < mutation_rate * 0.1:  # 降低阶段数量变异的概率
            new_stages = random.choice([s for s in self.search_space["stages"] if s <= len(mutated_config["stages"]) + 1])
            if new_stages > len(mutated_config["stages"]):
                # 添加阶段
                current_channels = mutated_config["stages"][-1]["channels"] if mutated_config["stages"] else mutated_config["input_channels"]
                for _ in range(new_stages - len(mutated_config["stages"])):
                    new_stage = self._generate_random_stage(current_channels)
                    mutated_config["stages"].append(new_stage)
            else:
                # 删除阶段
                mutated_config["stages"] = mutated_config["stages"][:new_stages]
        
        # 变异每个阶段
        for stage in mutated_config["stages"]:
            if random.random() < mutation_rate:
                stage["channels"] = random.choice(self.search_space["channels"])
            
            # 变异 block 数量
            if random.random() < mutation_rate * 0.4:
                new_blocks = random.choice([b for b in self.search_space["blocks_per_stage"] if b <= len(stage["blocks"]) + 1])
                if new_blocks > len(stage["blocks"]):
                    # 添加block
                    current_channels = stage["channels"]
                    for _ in range(new_blocks - len(stage["blocks"])):
                        new_block = self._generate_block(current_channels, stage["channels"], False)
                        stage["blocks"].append(new_block)
                else:
                    # 删除block
                    stage["blocks"] = stage["blocks"][:new_blocks]
            
            # 变异每个block
            for block in stage["blocks"]:
                if random.random() < mutation_rate:
                    old_type = block["type"]
                    block["type"] = random.choice(self.search_space["conv_types"])
                    
                    # 如果类型变为SeDpConv，需要调整通道数
                    if block["type"] == "SeDpConv" and old_type != "SeDpConv":
                        stage["channels"] = stage["channels"]  # 保持通道数不变

                    # 根据新的block类型更新skip connection
                    if block["type"] in ["DpConv", "SeDpConv", "SeSepConv"]:
                        block["skip_connection"] = False
                    else:
                        block["skip_connection"] = random.choice(self.search_space["skip_connection"])
                
                if random.random() < mutation_rate:
                    block["kernel_size"] = random.choice(self.search_space["kernel_sizes"])
                
                if random.random() < mutation_rate:
                    new_expansion = random.choice([e for e in self.search_space["expansions"] if e <= 3])
                    # 处理MBConv和DWSepConv的关系
                    if block["type"] == "MBConv" and new_expansion == 1:
                        block["type"] = "DWSepConv"
                        # 更新skip connection
                        block["skip_connection"] = random.choice(self.search_space["skip_connection"])
                    elif block["type"] == "DWSepConv" and new_expansion > 1:
                        block["type"] = "MBConv"
                        # 更新skip connection
                        block["skip_connection"] = random.choice(self.search_space["skip_connection"])
                    else:
                        block["expansion"] = new_expansion
                
                if random.random() < mutation_rate:
                    block["has_se"] = random.choice(self.search_space["has_se"])
                    # 同步更新se_ratio
                    if block["has_se"]:
                        block["se_ratio"] = random.choice([r for r in self.search_space["se_ratios"] if r > 0])
                    else:
                        block["se_ratio"] = 0
                
                if random.random() < mutation_rate and block["type"] not in ["DpConv", "SeDpConv", "SeSepConv"]:
                    # 只有支持 skip connection 的 block 才能变异 skip_connection
                    block["skip_connection"] = random.choice(self.search_space["skip_connection"])

                if random.random() < mutation_rate:
                    block["activation"] = random.choice(self.search_space["activations"])
        
        return mutated_config
    
    def _generate_random_stage(self, in_channels: int) -> Dict[str, Any]:
        """生成随机阶段配置"""
        stage_channels = random.choice(self.search_space["channels"])
        num_blocks = random.choice(self.search_space["blocks_per_stage"])
        max_blocks = min(2, max(self.search_space["blocks_per_stage"]))
        num_blocks = random.choice([b for b in self.search_space["blocks_per_stage"] if b <= max_blocks])
        
        blocks = []
        current_channels = in_channels
        for i in range(num_blocks):
            block = self._generate_block(current_channels, stage_channels, i == 0)
            blocks.append(block)
            if block["type"] == "SeDpConv":
                current_channels = stage_channels
            else:
                current_channels = stage_channels
        
        return {
            "blocks": blocks,
            "channels": stage_channels
        }
    
    def _generate_random_block(self) -> Dict[str, Any]:
        """生成随机block配置（用于散射生成器）"""
        # 随机选择输入输出通道
        in_channels = random.choice(self.search_space["channels"])
        stage_channels = random.choice(self.search_space["channels"])
        return self._generate_block(in_channels, stage_channels, False)


class ScatteringGenerator:
    """散射生成器 - 生成多样化的初始配置"""
    
    def __init__(self, config_generator: ConfigGenerator):
        self.config_generator = config_generator
        # 为什么需要这个themes?
        self.scattering_themes = ["efficient", "accurate", "balanced", "tiny", "fast"]
    
    def generate_scattered_seeds(self, num_seeds: int) -> List[Dict[str, Any]]:
        """生成散射的初始种子配置"""
        seeds = []
        
        for i in range(num_seeds):
            theme = self.scattering_themes[i % len(self.scattering_themes)]
            config = self._generate_theme_config(theme)
            seeds.append(config)
        
        return seeds
    
    def _generate_theme_config(self, theme: str) -> Dict[str, Any]:
        """根据主题生成配置"""
        base_config = self.config_generator.generate_random_config()
        
        if theme == "efficient":
            # 偏向高效的小模型
            return self._make_config_efficient(base_config)
        elif theme == "accurate":
            # 偏向准确率的大模型
            return self._make_config_accurate(base_config)
        elif theme == "balanced":
            # 平衡配置
            return base_config
        elif theme == "tiny":
            # 极小模型
            return self._make_config_tiny(base_config)
        elif theme == "fast":
            # 快速推理模型
            return self._make_config_fast(base_config)
        else:
            return base_config
    
    def _make_config_efficient(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使配置更高效"""
        # 减少通道数
        for stage in config["stages"]:
            # 检查是否有 SeDpConv block
            has_se_dp_conv = any(block.get("type") == "SeDpConv" for block in stage["blocks"])
            
            if has_se_dp_conv:
                # 如果有 SeDpConv， channels 必须等于 input_channels 或前一个 stage 的 channels
                # 对于第一个stage，等于input_channels；对于后续 stage，等于前一个 stage 的 channels
                stage_idx = config["stages"].index(stage)
                if stage_idx == 0:
                    stage["channels"] = config["input_channels"]
                else:
                    stage["channels"] = config["stages"][stage_idx-1]["channels"]
            else:
                # 否则可以减少通道数
                stage["channels"] = max(8, stage["channels"] // 2)
                
            # 减少block数量
            if len(stage["blocks"]) > 1:
                stage["blocks"] = stage["blocks"][:1]
        return config
    
    def _make_config_accurate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使配置更准确"""
        # 增加通道数
        for stage in config["stages"]:
            # 检查是否有 SeDpConv block
            has_se_dp_conv = any(block.get("type") == "SeDpConv" for block in stage["blocks"])
            
            if not has_se_dp_conv:
                # 只有非 SeDpConv 的stage才能增加通道数
                stage["channels"] = min(64, stage["channels"] * 2)
                
            # 增加block数量
            if len(stage["blocks"]) < 3:
                new_block = self.config_generator._generate_random_block()
                new_block["stride"] = 1  # 确保不是第一个 block
                stage["blocks"].append(new_block)
        return config
    
    def _make_config_tiny(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使配置极小"""
        # 最小化配置
        config["stages"] = config["stages"][:1]  # 只保留一个阶段
        for stage in config["stages"]:
            # 检查是否有 SeDpConv block
            has_se_dp_conv = any(block.get("type") == "SeDpConv" for block in stage["blocks"])
            
            if has_se_dp_conv:
                # 如果有 SeDpConv， channels 必须等于 input_channels
                stage["channels"] = config["input_channels"]
            else:
                # 否则可以设置为最小通道数
                stage["channels"] = 8
                
            stage["blocks"] = stage["blocks"][:1]  # 只保留一个block
            for block in stage["blocks"]:
                block["expansion"] = 1
                block["has_se"] = False
        return config
    
    def _make_config_fast(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使配置更快"""
        # 使用更小的kernel和更少的计算
        for stage in config["stages"]:
            for block in stage["blocks"]:
                block["kernel_size"] = 3
                block["expansion"] = 1
                if block["type"] in ["MBConv", "SeSepConv"]:
                    block["type"] = "DpConv"  # 使用更简单的卷积类型
        return config