# models/scattered_forest_search.py
import uuid
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .mcts_nodes import MCTSNode, MCTSTree
from .config_generator import ConfigGenerator, ScatteringGenerator
from .evaluator import ModelEvaluator
from models import CandidateModel
from llm_prompt import LLMConfigGenerator
from utils import calculate_memory_usage
from data import get_dataset_info, get_multitask_dataloaders


class ScatteredForestSearch:
    """散射森林搜索算法"""
    
    def __init__(self, search_space: Dict[str, Any], constraints: Dict[str, float], dataset_name: str,
                 device: str = "cuda", exploration_weight: float = 1.414):
        self.search_space = search_space
        self.constraints = constraints
        self.device = device
        self.exploration_weight = exploration_weight
        
        self.dataset_info = get_dataset_info(dataset_name)

        multitask_dataloaders = get_multitask_dataloaders(root_dir="/root/har_train/data/UniMTS_data", datasets=[dataset_name])
        # 检查是否成功加载了指定数据集
        if dataset_name not in multitask_dataloaders:
            available_datasets = list(multitask_dataloaders.keys())
            raise ValueError(f"数据集 {dataset_name} 加载失败。可用的数据集: {available_datasets}")
        # 提取单个数据集的数据加载器
        self.dataloader = multitask_dataloaders[dataset_name]
        print(f"✅ 成功加载数据集 {dataset_name}:")
        print(f"   - 训练集: {len(self.dataloader['train'].dataset)} 样本")
        print(f"   - 测试集: {len(self.dataloader['test'].dataset)} 样本")

        # 新增内存约束相关属性
        self.max_memory = float(constraints.get("max_peak_memory", 20e6))/1e6  # MB
        print(f"max memory: {self.max_memory}MB")

        # 初始化组件
        self.config_generator = ConfigGenerator(search_space, self.dataset_info, self.max_memory)
        self.scattering_generator = ScatteringGenerator(self.config_generator)
        self.evaluator = ModelEvaluator(constraints, device, dataloader=self.dataloader)
        self.llm_config_generator = LLMConfigGenerator(search_space, constraints, dataset_name)  # 新增

        # 搜索状态 - 使用单个 MCTSTree （支持森林）
        self.tree = MCTSTree(exploration_weight=exploration_weight)
        self.best_candidate: Optional[CandidateModel] = None
        self.best_reward: float = -float('inf')
        self.iteration_count: int = 0

        # 量化方向选项
        self.quant_directions = ["none", "static", "qat"]

        self.max_retry_attempts = 3  # 最大重试次数

        # 新增：去重相关属性
        self.seen_configs = set()  # 存储已见过的配置哈希值
        self.duplicate_count = 0   # 重复配置计数

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """生成配置的哈希值用于去重"""
        # 创建一个规范化的配置副本，移除可能变化的字段
        normalized_config = self._normalize_config(config)
        
        # 将配置转换为JSON字符串并生成哈希
        config_str = json.dumps(normalized_config, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化配置，移除不影响模型结构的字段"""
        normalized = {
            "input_channels": config.get("input_channels"),
            "num_classes": config.get("num_classes"),
            "stages": []
        }
        
        # 处理每个stage
        for stage in config.get("stages", []):
            normalized_stage = {
                "channels": stage.get("channels"),
                "blocks": []
            }
            
            # 处理每个block
            for block in stage.get("blocks", []):
                normalized_block = {
                    "type": block.get("type"),
                    "kernel_size": block.get("kernel_size"),
                    "stride": block.get("stride"),
                    "expansion": block.get("expansion"),
                    "has_se": block.get("has_se"),
                    "se_ratio": block.get("se_ratio"),
                    "skip_connection": block.get("skip_connection"),
                    "activation": block.get("activation")
                }
                normalized_stage["blocks"].append(normalized_block)
            
            normalized["stages"].append(normalized_stage)
        
        return normalized
    
    def _is_duplicate_config(self, config: Dict[str, Any]) -> bool:
        """检查配置是否重复"""
        config_hash = self._generate_config_hash(config)
        return config_hash in self.seen_configs
    
    def _add_config_to_seen(self, config: Dict[str, Any]):
        """将配置添加到已见集合"""
        config_hash = self._generate_config_hash(config)
        self.seen_configs.add(config_hash)
        
    def initialize_forest(self, num_seeds: int = 5):
        """初始化森林"""
        print(f"初始化森林: {num_seeds} 个种子")

        # 清空已见配置集合
        self.seen_configs.clear()
        self.duplicate_count = 0
        
        # 生成散射的种子配置
        scattered_seeds = self.scattering_generator.generate_scattered_seeds(num_seeds)
        
        for i, seed_config in enumerate(scattered_seeds):
            # 检查是否重复
            if self._is_duplicate_config(seed_config):
                print(f"跳过重复的种子配置 {i}")
                self.duplicate_count += 1
                continue
            print(f"seed config:\n {seed_config}")
            candidate = CandidateModel(config=seed_config)
            # 评估种子 reward就是准确率（0-100范围）
            reward, metrics = self.evaluator.evaluate_candidate(candidate)

            # 创建节点
            node_id = f"seed_{i}"
            node = MCTSNode(
                node_id=node_id,
                candidate=candidate,
                directions=self.quant_directions
            )
            # 这个地方除了增加 visit，还累加了 total reward
            node.update_reward(reward)
            node.is_forest_root = True

            # 添加到森林作为根节点
            self.tree.add_node(node, is_forest_root=True)
            
            # 为种子节点生成散射方向
            scattering_directions = self.tree.scattering(node)
            node.directions = scattering_directions

            # 初始化方向统计
            for direction in scattering_directions:
                node.direction_q_values[direction] = 0.0
                node.direction_visits[direction] = 0

            # 更新最佳候选
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = candidate
            
            print(f"种子 {node_id}: 奖励 = {reward:.4f}, "
                  f"准确率 = {metrics.get('accuracy', 0):.4f}")
    
    def _check_memory_constraint(self, config: Dict[str, Any]) -> Tuple[bool, float, str]:
        """检查配置的内存使用情况"""
        try:
            candidate = CandidateModel(config=config)
            model = candidate.build_model()
            # 计算内存使用量
            memory_info = calculate_memory_usage(model, input_size=(64, self.dataset_info['channels'], self.dataset_info['time_steps']), device='cuda')

            
            memory_usage = memory_info["total_memory_MB"]

            # 根据量化模式调整内存使用量
            quant_mode = config.get("quant_mode", "none")
            if quant_mode in ["static", "qat"]:
                # 量化模型通常可以压缩到原来的 1/4 左右
                compressed_memory = memory_usage / 4.0
                print(f"量化模型内存压缩: {memory_usage:.2f}MB → {compressed_memory:.2f}MB (quant_mode: {quant_mode})")
                memory_usage = compressed_memory

            # 检查是否超过限制
            if memory_usage <= self.max_memory:
                return True, memory_usage, "OK"
            else:
                error_msg = f"内存使用 {memory_usage:.2f}MB 超过限制 {self.max_memory}MB"
                return False, memory_usage, error_msg
                
        except Exception as e:
            print(f"内存计算失败: {e}")
            return False, 0, f"内存计算失败: {str(e)}"

    def search(self, iterations: int = 100, exploration_weight: float = 1.0,
               dataset_names: list = None):
        """执行搜索"""
        print(f"开始 SFS 搜索: {iterations} 次迭代")
        
        for iteration in range(iterations):
            self.iteration_count += 1

            # 1. 选择种子节点 (Foresting)
            selected_seed = self.tree.select_forest_root()
            if not selected_seed:
                print("没有可用的种子节点，重新初始化森林")
                self.initialize_forest(3)
                selected_seed = self.tree.select_forest_root()

            # 2. 从种子节点开始模拟， 选择要扩展的节点
            current_node, trajectory = self._simulate_from_seed(selected_seed)

            # 3. Scattering 选择优化方向
            direction = current_node.get_best_direction(self.exploration_weight)

            # 4. 使用 LLM 配置生成器生成新配置
            new_config = self._generate_config_with_llm(current_node, direction)

            # 检查是否重复
            if self._is_duplicate_config(new_config):
                print(f"🔁 迭代 {iteration}: 跳过重复配置")
                self.duplicate_count += 1
                continue

            # 5. 创建新候选并评估
            new_candidate = CandidateModel(config=new_config)
            reward, metrics = self.evaluator.evaluate_candidate(new_candidate, dataset_names)
            
            # 6. 扩展节点
            child_node = self._create_child_node(current_node, direction, new_candidate, reward)
            
            # 7. 反向传播奖励
            self._backpropagate(trajectory + [(current_node, direction, child_node)], reward)
            
            # 8. Scouting: 更新全局经验
            feedback = {
                "reward": reward,
                "accuracy": metrics.get('accuracy', 0),
                "direction": direction,
                "parent_config": current_node.candidate.config if current_node.candidate else {},
                "child_config": new_config
            }
            self.tree.scouting(current_node, direction, child_node, reward, feedback)

            # 9. 为子节点生成新的散射方向
            scattering_directions = self.tree.scattering(child_node)
            child_node.directions = scattering_directions
            
            # 添加到已见配置
            self._add_config_to_seen(new_config)

            # 更新最佳候选
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = new_candidate
                print(f"🎯 迭代 {iteration}: 发现新的最佳候选! "
                      f"奖励 = {reward:.4f}, 准确率 = {metrics.get('accuracy', 0):.4f}")
            
            if iteration % 10 == 0:
                self._print_search_progress(iteration)
    
    def _select_seed_node(self, tree: MCTSTree, exploration_weight: float) -> MCTSNode:
        """选择种子节点"""
        # 简单的策略：选择访问次数最少的节点进行探索
        min_visits = float('inf')
        selected_node = None
        
        for node in tree.nodes.values():
            if node.visit_count < min_visits:
                min_visits = node.visit_count
                selected_node = node
        
        return selected_node or list(tree.nodes.values())[0]
    
    def _simulate_from_seed(self, seed_node: MCTSNode) -> Tuple[MCTSNode, List]:
        """从种子节点开始模拟，选择要扩展的节点"""
        trajectory = []
        current_node = seed_node
        
        # 模拟直到找到叶子节点或达到最大深度
        max_depth = 5
        depth = 0
        
        while depth < max_depth and current_node.children:
            # 使用 UCT 选择最佳方向
            direction = current_node.get_best_direction(self.exploration_weight)
            
            # 如果该方向有子节点，继续前进
            if direction in current_node.children:
                next_node = current_node.children[direction]
                trajectory.append((current_node, direction, next_node))
                current_node = next_node
                depth += 1
            else:
                # 没有子节点，选择扩展这个节点
                break
        
        return current_node, trajectory
    
    def _generate_config_with_llm(self, parent_node: MCTSNode, direction: str) -> Dict[str, Any]:
        """使用 LLM 配置生成器生成新配置"""
        parent_config = parent_node.candidate.config if parent_node.candidate else {}
        
        parent_performance = {
            "average_reward": parent_node.average_reward,
            "visit_count": parent_node.visit_count,
            "directions_explored": list(parent_node.direction_visits.keys())
        }
        
        # 修复：传递全局经验给 LLM
        global_insights = self.tree.global_insights

        # 重试机制
        for attempt in range(self.max_retry_attempts + 1):
            try:
                # 生成配置
                new_config = self.llm_config_generator.generate_config_with_context(
                    parent_config, direction, parent_performance, global_insights,
                    memory_feedback=None if attempt == 0 else memory_feedback
                )
                print(f"new config:\n {new_config}")
                # 检查内存约束
                memory_ok, memory_usage, memory_msg = self._check_memory_constraint(new_config)
                print(f"memory usage: {memory_usage}MB")
                # 检查是否重复
                if self._is_duplicate_config(new_config):
                    print(f"🔄 LLM 生成重复配置 (尝试 {attempt+1}/{self.max_retry_attempts})，重新生成...")
                    memory_feedback = f"""
                    The previous model config: {json.dumps(new_config)}
                    The generated configuration is a duplicate of a previously seen model. 
                    Please generate a different architecture. This is attempt {attempt+1}/{self.max_retry_attempts}.
                    Suggestions:
                    - Change the number of stages
                    - Modify the number of blocks in stages
                    - Use different convolution types
                    - Adjust channel numbers
                    - Try different expansion ratios
                    """
                    continue

                if memory_ok:
                    print(f"✅ 配置通过内存检查: {memory_usage:.2f}MB")
                    return new_config
                else:
                    print(f"⚠️ 配置内存超标 ({attempt+1}/{self.max_retry_attempts}): {memory_msg}")
                    
                    # 如果是最后一次尝试，使用智能降级
                    if attempt == self.max_retry_attempts:
                        print("🚨 达到最大重试次数，使用智能降级配置")
                        return self._generate_degraded_config(parent_config, direction, memory_usage)
                    
                    # 更新提示词，加入内存反馈
                    memory_feedback = f"""
                    The previous model config: {json.dumps(new_config)}
                    The generated model configuration memory usage is {memory_usage:.2f}MB, exceeding the maximum limit {self.max_memory}MB.
                    Please generate a lighter configuration. This is the {attempt}/{self.max_retry_attempts} retry. 
                    Suggestions:
                    - Reduce the number of stages
                    - Reduce the number of blocks per phase (blocks)
                    - Use simpler convolution types
                    - Close se module or reduce expansion ratio
                    """
                    
            except Exception as e:
                print(f"LLM configuration generation failed (Attempt {attempt+1}): {e}")
                if attempt == self.max_retry_attempts:
                    return self._generate_degraded_config(parent_config, direction, 0)
        # 最终回退
        return self._generate_degraded_config(parent_config, direction, 0)
    
    def _generate_degraded_config(self, base_config: Dict[str, Any], direction: str, 
                            current_memory: float) -> Dict[str, Any]:
        """生成降级配置以确保内存安全 - 基于卷积类型的内存开销优化"""
        print("🛠️ 使用智能降级生成安全配置")
        
        # 基于基础配置创建安全版本
        safe_config = base_config.copy() if base_config else {}
        safe_config["quant_mode"] = direction
        
        # 如果没有基础配置，生成一个最小配置
        if not safe_config:
            safe_config = {
                "input_channels": self.search_space["input_channels"],
                "num_classes": self.search_space["num_classes"], 
                "quant_mode": direction,
                "stages": [
                    {
                        "blocks": [
                            {
                                "type": "SeDpConv",  # 使用内存最小的卷积类型
                                "kernel_size": 3,
                                "stride": 1,
                                "expansion": 1,
                                "has_se": False,
                                "activation": "ReLU"
                            }
                        ],
                        "channels": 8
                    }
                ]
            }
            # 检查是否重复
            if not self._is_duplicate_config(safe_config):
                return safe_config
        
        # 尝试多种降级策略，同时避免重复
        strategies = [
            self._apply_stage_reduction,
            self._apply_conv_type_optimization,
            self._apply_channel_reduction,
            self._apply_block_reduction
        ]
        
        for strategy in strategies:
            temp_config = strategy(safe_config.copy(), current_memory)
            if temp_config and not self._is_duplicate_config(temp_config):
                memory_ok, memory_usage, _ = self._check_memory_constraint(temp_config)
                if memory_ok:
                    return temp_config
                
        # 最终手段 - 生成绝对最小配置
        print("🚨 策略5: 使用绝对最小配置")
        minimal_config = {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"], 
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "SeDpConv",  # 内存最小的卷积类型
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "activation": "ReLU"
                        }
                    ],
                    "channels": 8  # 最小通道数
                }
            ]
        }

        # 如果最小配置也重复，稍微修改一下
        attempt = 0
        while self._is_duplicate_config(minimal_config) and attempt < 10:
            # 轻微修改配置以避免重复
            minimal_config["stages"][0]["channels"] += 1
            attempt += 1
        
        return minimal_config

        
    def _apply_stage_reduction(self, config: Dict[str, Any], current_memory: float) -> Optional[Dict[str, Any]]:
        # 策略1: 如果内存严重超标，先减少 stage 数量
        if current_memory <= self.max_memory:
            return None  # 如果已经满足内存要求，不需要减少stage
        
        original_stage_count = len(config.get("stages", []))
        if original_stage_count <= 1:
            return None  # 至少保留一个stage
        
        print("🔧 策略1: 减少 stage 数量")

        # 从当前配置开始，逐步减少stage数量
        temp_config = config.copy()
        stage_count = original_stage_count
        
        while stage_count > 1:
            # 减少一个stage
            temp_config["stages"] = temp_config["stages"][:-1]
            stage_count -= 1
            
            # 检查内存是否满足
            memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
            
            if memory_ok:
                print(f"✅ 减少 stage 数量: {original_stage_count} → {stage_count}, 内存: {new_memory:.2f}MB")
                return temp_config
            else:
                print(f"⚠️ 减少 stage 数量: {original_stage_count} → {stage_count}, 内存: {new_memory:.2f}MB (仍超标)")
                
                # 如果只剩下一个stage还是不满足，就停止
                if stage_count == 1:
                    break
        
        # 如果所有尝试都不成功，返回None
        return None
            
    def _apply_conv_type_optimization(self, config: Dict[str, Any], current_memory: float) -> Optional[Dict[str, Any]]:
        # 策略2: 替换卷积类型 - 持续替换直到满足内存要求
        if current_memory <= self.max_memory:
            return None  # 如果已经满足内存要求，不需要优化
        
        print("🔧 策略2: 优化卷积类型")
        # 卷积类型的内存开销排序（从大到小）
        conv_type_priority = ["MBConv", "DWSepConv", "SeSepConv", "DpConv", "SeDpConv"]

        # 创建配置副本进行操作
        temp_config = json.loads(json.dumps(config))  # 深拷贝

        # 从最后一个stage开始处理（通常后面的stage参数更多）
        for stage_idx in range(len(temp_config.get("stages", [])) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            
            # 对当前stage的所有block进行优化（从后往前）
            for block_idx in range(len(stage.get("blocks", [])) - 1, -1, -1):
                block = stage["blocks"][block_idx]
                current_conv_type = block.get("type", "MBConv")
                
                # 如果当前类型已经是最小的，跳过
                if current_conv_type == "SeDpConv":
                    continue
                    
                # 找到当前类型在优先级中的位置
                try:
                    current_priority = conv_type_priority.index(current_conv_type)
                except ValueError:
                    current_priority = 0
                    
                # 记录原始类型
                original_type = block["type"]
                
                # 尝试更小的卷积类型
                found_improvement = False
                for smaller_type in conv_type_priority[current_priority + 1:]:
                    # 替换为更小的类型
                    block["type"] = smaller_type
                    
                    # 检查内存
                    memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
                    
                    if memory_ok:
                        print(f"✅ Stage{stage_idx}-Block{block_idx}: {original_type} → {smaller_type}, 内存: {new_memory:.2f}MB")
                        return temp_config
                    else:
                        print(f"⚠️ Stage{stage_idx}-Block{block_idx}: {original_type} → {smaller_type}, 内存: {new_memory:.2f}MB (仍超标)")
                        # 保留这个替换，继续尝试其他block
                        found_improvement = True
                        break  # 找到一个可替换的类型就继续，不尝试更小的类型
                
                # 如果没有找到任何可替换的类型，恢复原始类型
                if not found_improvement:
                    block["type"] = original_type
                
                # 检查当前状态是否满足内存要求（累积替换的效果）
                memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
                if memory_ok:
                    print(f"✅ 经过多次替换后内存达标: {new_memory:.2f}MB")
                    return temp_config
        
        # 如果所有替换尝试都不成功，返回None
        return None
    
    def _apply_channel_reduction(self, config: Dict[str, Any], current_memory: float) -> Optional[Dict[str, Any]]:
        # 策略3: 如果还是超标，减少通道数（从最后一个stage开始）
        print("🔧 策略3: 减少通道数")
        for stage_idx in range(len(config.get("stages", [])) - 1, -1, -1):
            stage = config["stages"][stage_idx]
            original_channels = stage["channels"]
            
            # 尝试逐步减少通道数
            for reduction_factor in [0.75, 0.5, 0.25]:
                new_channels = max(8, int(original_channels * reduction_factor))
                if new_channels == stage["channels"]:
                    continue
                    
                stage["channels"] = new_channels
                
                # 检查内存
                memory_ok, new_memory, _ = self._check_memory_constraint(config)
                
                if memory_ok:
                    print(f"✅ Stage{stage_idx}通道数: {original_channels} → {new_channels}, 内存: {new_memory:.2f}MB")
                    return config
                else:
                    print(f"⚠️ Stage{stage_idx}通道数: {original_channels} → {new_channels}, 内存: {new_memory:.2f}MB (仍超标)")
            
            # 恢复原始通道数
            stage["channels"] = original_channels
        return None
    
    def _apply_block_reduction(self, config: Dict[str, Any], current_memory: float) -> Optional[Dict[str, Any]]:
        """应用block减少策略 - 激进减少直到满足内存要求"""
        if current_memory <= self.max_memory:
            return None
            
        print("🔧 策略4: 减少 block 数量")
        
        # 创建配置副本进行操作
        temp_config = json.loads(json.dumps(config))
        
        # 记录原始配置信息
        original_stages = len(temp_config["stages"])
        
        # 从最后一个stage开始处理
        for stage_idx in range(len(temp_config["stages"]) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            original_block_count = len(stage["blocks"])
            
            if original_block_count <= 1:
                # 如果这个stage只有一个block，考虑删除整个stage
                if stage_idx > 0:  # 不能删除第一个stage
                    print(f"🔄 Stage{stage_idx} 只有一个block，尝试删除整个stage")
                    deleted_stage = temp_config["stages"].pop(stage_idx)
                    
                    # 检查内存
                    memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
                    
                    if memory_ok:
                        print(f"✅ 删除 Stage{stage_idx} 后内存达标: {new_memory:.2f}MB")
                        return temp_config
                    else:
                        print(f"⚠️ 删除 Stage{stage_idx} 后内存仍超标: {new_memory:.2f}MB")
                        # 保留删除结果，继续尝试
                        continue
                else:
                    continue  # 第一个stage不能删除
            
            # 逐步减少当前stage的block数量
            for new_block_count in range(original_block_count - 1, 0, -1):
                stage["blocks"] = stage["blocks"][:new_block_count]
                
                # 检查内存
                memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
                
                if memory_ok:
                    print(f"✅ Stage{stage_idx} block数: {original_block_count} → {new_block_count}, 内存: {new_memory:.2f}MB")
                    return temp_config
                else:
                    print(f"⚠️ Stage{stage_idx} block数: {original_block_count} → {new_block_count}, 内存: {new_memory:.2f}MB (仍超标)")
                    
                    # 如果减少到1个block还不满足，考虑删除这个stage
                    if new_block_count == 1 and stage_idx > 0:
                        print(f"🔄 Stage{stage_idx} 减少到1个block仍不满足，尝试删除整个stage")
                        deleted_stage = temp_config["stages"].pop(stage_idx)
                        
                        # 检查内存
                        memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
                        
                        if memory_ok:
                            print(f"✅ 删除 Stage{stage_idx} 后内存达标: {new_memory:.2f}MB")
                            return temp_config
                        else:
                            print(f"⚠️ 删除 Stage{stage_idx} 后内存仍超标: {new_memory:.2f}MB")
                            # 保留删除结果，继续处理其他stage
                            break  # 跳出当前stage的循环，处理下一个stage
        
        # 最终检查：如果经过所有操作后满足要求
        memory_ok, new_memory, _ = self._check_memory_constraint(temp_config)
        if memory_ok:
            print(f"✅ 最终减少后内存达标: {new_memory:.2f}MB")
            return temp_config
        
        # 如果所有尝试都失败，尝试极端情况：只保留第一个stage的第一个block
        if len(temp_config["stages"]) > 1 or len(temp_config["stages"][0]["blocks"]) > 1:
            print("🚨 尝试极端情况：只保留第一个stage的第一个block")
            minimal_config = {
                "input_channels": temp_config["input_channels"],
                "num_classes": temp_config["num_classes"],
                "quant_mode": temp_config["quant_mode"],
                "stages": [
                    {
                        "blocks": [temp_config["stages"][0]["blocks"][0]],
                        "channels": temp_config["stages"][0]["channels"]
                    }
                ]
            }
            
            memory_ok, new_memory, _ = self._check_memory_constraint(minimal_config)
            if memory_ok:
                print(f"✅ 极端简化后内存达标: {new_memory:.2f}MB")
                return minimal_config
        
        return None
        
        # # 验证最小配置的内存
        # memory_ok, final_memory, _ = self._check_memory_constraint(minimal_config)
        # if memory_ok:
        #     print(f"✅ 最小配置内存达标: {final_memory:.2f}MB")
        #     return minimal_config
        # else:
        #     print(f"❌ 警告: 即使最小配置也超标: {final_memory:.2f}MB")
        #     return minimal_config  # 仍然返回，让评估器处理
    
    def _create_child_node(self, parent_node: MCTSNode, direction: str, 
                          candidate: CandidateModel, reward: float) -> MCTSNode:
        """创建子节点"""
        child_node_id = f"iter_{self.iteration_count}_dir_{direction}"
        child_node = MCTSNode(
            node_id=child_node_id,
            candidate=candidate,
            directions=parent_node.directions.copy()
        )
        child_node.update_reward(reward)
        
        # 添加到树中
        self.tree.add_node(child_node, parent_node, direction)
        
        return child_node
    
    def _backpropagate(self, trajectory: List, reward: float):
        """反向传播奖励"""
        # 沿着轨迹向上传播
        current_reward = reward

        for i, (node, direction, next_node) in enumerate(reversed(trajectory)):
            # 更新方向统计
            node.update_direction_stats(direction, current_reward)
            
            # 使用 max 策略更新 Q 值（论文公式2）
            current_q = node.direction_q_values.get(direction, 0.0)
            
            # 获取下一节点的最大 Q 值
            next_max_q = 0.0
            if next_node and next_node.direction_q_values:
                next_max_q = max(next_node.direction_q_values.values())
            
            new_q = max(current_q, next_max_q)
            
            # 加权平均更新（论文公式2）
            visits = node.direction_visits.get(direction, 0)
            alpha = 1.0 / (visits + 1) if visits > 0 else 1.0
            updated_q = (1 - alpha) * current_q + alpha * new_q
            
            node.direction_q_values[direction] = updated_q
            
            # 同时更新节点的总奖励（带衰减）
            node.update_reward(current_reward)
            current_reward = node.average_reward * 0.9  # 衰减因子
    
    def _print_search_progress(self, iteration: int):
        """打印搜索进度"""
        stats = self.tree.get_graph_statistics()
        
        print(f"\n--- 迭代 {iteration} 进度报告 ---")
        print(f"最佳奖励: {self.best_reward:.4f}")
        print(f"总节点数: {stats['total_nodes']}")
        print(f"评估节点: {stats['evaluated_nodes']}")
        print(f"森林大小: {stats['forest_count']} 棵树")
        print(f"平均奖励: {stats['average_reward']:.4f}")
        print(f"全局经验数: {stats['global_insights_count']}")
        print("------------------------------\n")

        # 显示全局经验
        print("全局经验:")
        for direction, insight in self.tree.global_insights.items():
            success_rate = insight.get('success_rate', 0)
            avg_reward = insight.get('average_reward', 0)
            print(f"  {direction}: 成功率={success_rate:.3f}, 平均奖励={avg_reward:.3f}")
        
        print("------------------------------\n")
    
    def get_best_model(self) -> Optional[CandidateModel]:
        """获取最佳模型"""
        return self.best_candidate
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        stats = self.tree.get_graph_statistics()
        
        return {
            "iterations": self.iteration_count,
            "best_reward": self.best_reward,
            "tree_statistics": stats,
            "global_insights": self.tree.global_insights,
            "unique_configs": len(self.seen_configs),
            "duplicate_count": self.duplicate_count
        }
    
    def save_search_state(self, filepath: str):
        """保存搜索状态"""
        self.tree.save_graph_info(filepath)

def create_sfs_search(search_space: Dict[str, Any], constraints: Dict[str, float], 
                     device: str = "cuda", exploration_weight: float = 1.414, dataset_name: str = "MMAct") -> ScatteredForestSearch:
    """创建SFS搜索实例的工厂函数"""
    return ScatteredForestSearch(
        search_space=search_space,
        constraints=constraints,
        device=device,
        exploration_weight=exploration_weight,
        dataset_name=dataset_name
    )