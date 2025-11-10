"""
基于队列的多GPU并行搜索 - 保留完整MCTS逻辑

架构:
  主进程: 维护搜索树，生成配置，协调全局状态
  工作进程: 在各个GPU上训练评估模型
"""
import uuid
import json
import hashlib
import numpy as np
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from queue import Empty
import time
# 导入你的原始类
from mcts.mcts_nodes import MCTSNode, MCTSTree
from mcts.config_generator import ConfigGenerator, ScatteringGenerator
from models import CandidateModel
from llm_prompt import LLMConfigGenerator
from utils import calculate_memory_usage
from data import get_dataset_info, get_multitask_dataloaders
from mcts import worker_process
# ✅ 导入降级策略管理器
from mcts import ConfigDegradationManager


class ParallelScatteredForestSearch:
    """
    并行散射森林搜索 - 主进程版本
    
    主进程负责:
      - 维护搜索树
      - 生成配置（LLM调用）
      - 协调全局状态
      - 分配任务到工作进程
    
    工作进程负责:
      - 训练评估模型
    """
    
    def __init__(self, search_space: Dict[str, Any], constraints: Dict[str, float],
                 dataset_name: str, num_gpus: int = 4, device: str = "cuda",
                 exploration_weight: float = 1.414, train_epochs: int = 100):
        self.search_space = search_space
        self.constraints = constraints
        self.dataset_name = dataset_name
        self.num_gpus = num_gpus
        self.device = device
        self.exploration_weight = exploration_weight
        self.train_epochs = train_epochs
        
        # 数据集信息
        self.dataset_info = get_dataset_info(dataset_name)
        
        # 内存约束
        self.max_memory = float(constraints.get("max_peak_memory", 20e6))/1e6
        
        # 配置生成器
        self.config_generator = ConfigGenerator(search_space, self.dataset_info, self.max_memory)
        self.scattering_generator = ScatteringGenerator(self.config_generator)
        self.llm_config_generator = LLMConfigGenerator(search_space, constraints, dataset_name)
        
        # ✅ 初始化降级策略管理器
        self.degradation_manager = ConfigDegradationManager(
            dataset_info=self.dataset_info,
            max_memory=self.max_memory,
            check_memory_fn=self._check_memory_constraint,
            is_duplicate_fn=self._is_duplicate_config
        )

        # 搜索树（主进程维护）
        self.tree = MCTSTree(exploration_weight=exploration_weight)
        self.best_candidate: Optional[CandidateModel] = None
        self.best_reward: float = -float('inf')
        self.iteration_count: int = 0
        
        # 量化方向
        self.quant_directions = ["none", "static", "qat", "qaft"]  # 添加qaft
        
        # 去重
        self.seen_configs = set()
        self.duplicate_count = 0
        self.max_retry_attempts = 3
        
        # 多进程组件
        self.task_queue = None
        self.result_queue = None
        self.workers = []
        
        print(f"✅ 并行搜索初始化: {num_gpus} GPUs")
        print(f"📋 量化模式: {', '.join(self.quant_directions)}")
    
    def _start_workers(self):
        """启动工作进程"""
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 创建队列
        self.task_queue = mp.Queue(maxsize=self.num_gpus * 2)
        self.result_queue = mp.Queue()
        
        # 启动工作进程
        for gpu_id in range(self.num_gpus):
            worker = mp.Process(
                target=worker_process,
                args=(gpu_id, self.task_queue, self.result_queue,
                      self.constraints, self.dataset_name, self.train_epochs)
            )
            worker.start()
            self.workers.append(worker)
            print(f"✅ 启动 Worker-GPU{gpu_id} (PID: {worker.pid})")
    
    def _stop_workers(self):
        """停止工作进程"""
        print("\n🛑 停止工作进程...")
        
        # 发送终止信号
        for _ in range(self.num_gpus):
            self.task_queue.put(None)
        
        # 等待进程结束
        for i, worker in enumerate(self.workers):
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"⚠️  Worker-GPU{i} 未响应，强制终止")
                worker.terminate()
                worker.join()
        
        print("✅ 所有工作进程已停止")
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """生成配置哈希"""
        normalized = self._normalize_config(config)
        config_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化配置"""
        # 与原版相同的实现
        normalized = {
            "input_channels": config.get("input_channels"),
            "num_classes": config.get("num_classes"),
            "stages": []
        }
        
        for stage in config.get("stages", []):
            normalized_stage = {
                "channels": stage.get("channels"),
                "blocks": []
            }
            
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
    
    def initialize_forest(self, num_seeds: int = 5):
        """初始化森林 - 使用工作进程池"""
        print(f"\n🌱 初始化森林: {num_seeds} 个种子")
        
        # 启动工作进程
        self._start_workers()
        
        # 生成种子配置, 每个种子对应一棵树，相当于是多棵树的森林，每棵树对应一个gpu
        scattered_seeds = self.scattering_generator.generate_scattered_seeds(num_seeds)
        
        # 提交所有种子到任务队列
        seed_ids = []
        for i, seed_config in enumerate(scattered_seeds):
            if self._is_duplicate_config(seed_config):
                print(f"跳过重复种子 {i}")
                continue
            
            seed_id = f"seed_{i}"
            seed_ids.append((seed_id, seed_config, i))
            self.task_queue.put((seed_id, seed_config))
            self._add_config_to_seen(seed_config)
        
        # 收集所有结果
        print(f"\n⏳ 等待 {len(seed_ids)} 个种子评估完成...")
        for seed_id, seed_config, i in seed_ids:
            # 从结果队列获取
            candidate_id, reward, metrics = self.result_queue.get()
            
            # 创建候选模型
            candidate = CandidateModel(config=seed_config)
            candidate.candidate_id = candidate_id
            candidate.metrics = metrics
            
            # 创建节点
            node = MCTSNode(
                node_id=seed_id,
                candidate=candidate,
                directions=self.quant_directions
            )
            node.update_reward(reward)
            node.is_forest_root = True
            
            # 添加到森林
            self.tree.add_node(node, is_forest_root=True)
            
            # 初始化方向
            scattering_directions = self.tree.scattering(node)
            node.directions = scattering_directions
            for direction in scattering_directions:
                node.direction_q_values[direction] = 0.0
                node.direction_visits[direction] = 0
            
            # 更新最佳
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = candidate
            
            print(f"✅ 种子 {seed_id}: 奖励={reward:.2f}, GPU={metrics.get('gpu_id')}")
    
    def search(self, iterations: int = 100, exploration_weight: float = 1.0, dataset_names: list = None):
        """执行搜索 - 使用工作进程池 流水线并行搜索"""
        print(f"\n🔍 开始搜索: {iterations} 次迭代")

        # ===== 阶段1: 预先生成一批任务填满GPU =====
        pending_tasks = {}  # {candidate_id: (current_node, direction, trajectory)}
        submitted_count = 0
        max_initial_retries = 10  # 防止无限循环

        while submitted_count < min(self.num_gpus, iterations) and max_initial_retries > 0:
            selected_seed = self.tree.select_forest_root()
            current_node, trajectory = self._simulate_from_seed(selected_seed)
            direction = current_node.get_best_direction(self.exploration_weight)
            
            new_config = self._generate_config_with_llm(current_node, direction)
            
            if self._is_duplicate_config(new_config):
                print(f"🔁 重复配置，重新生成 (已提交: {submitted_count}/{self.num_gpus})")
                max_initial_retries -= 1
                continue
            
            candidate_id = f"iter_{self.iteration_count}_dir_{direction}"
            self.iteration_count += 1
            submitted_count += 1
            
            self.task_queue.put((candidate_id, new_config))
            self._add_config_to_seen(new_config)
            
            pending_tasks[candidate_id] = (current_node, direction, trajectory, new_config)
            print(f"📤 提交任务 {candidate_id} ({submitted_count}/{min(self.num_gpus, iterations)})")

        # ===== 阶段2: 流水线执行 =====
        print(f"\n🔄 阶段2: 流水线执行")
        completed_count = 0

        while completed_count < iterations:
            # 🔵 非阻塞地获取结果
            try:
                result_id, reward, metrics = self.result_queue.get(timeout=1)
                print(f"✅ 收到结果 {result_id} (完成: {completed_count+1}/{iterations})")
                
                # 处理结果
                if result_id in pending_tasks:
                    current_node, direction, trajectory, new_config = pending_tasks.pop(result_id)
                    
                    # 创建子节点
                    new_candidate = CandidateModel(config=new_config)
                    new_candidate.candidate_id = result_id
                    new_candidate.metrics = metrics

                    # ✅ 新增：记录元信息
                    new_candidate.iteration = completed_count
                    new_candidate.parent_id = current_node.node_id
                    new_candidate.parent_direction = direction
                    new_candidate.root_seed_id = self._get_root_seed_id(current_node)
                    
                    child_node = MCTSNode(
                        node_id=result_id,
                        candidate=new_candidate,
                        directions=current_node.directions.copy()
                    )

                    # ✅ 记录迭代信息到节点
                    child_node.iteration = completed_count

                    child_node.update_reward(reward)
                    
                    # 添加到树
                    self.tree.add_node(child_node, current_node, direction)
                    
                    # 反向传播
                    self._backpropagate(trajectory + [(current_node, direction, child_node)], reward)
                    
                    # Scouting
                    feedback = {
                        "reward": reward,
                        "accuracy": metrics.get('accuracy', 0),
                        "direction": direction,
                        "parent_config": current_node.candidate.config,
                        "child_config": new_config
                    }
                    self.tree.scouting(current_node, direction, child_node, reward, feedback)
                    
                    # Scattering
                    child_node.directions = self.tree.scattering(child_node)
                    
                    # 更新最佳
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_candidate = new_candidate
                        print(f"🎯 新最佳! 奖励={reward:.2f}")
                    
                    completed_count += 1

                    # 每10个任务打印一次进度
                    if completed_count % 10 == 0:
                        self._print_search_progress(completed_count)
                
                # 🔵 立即生成新任务补充GPU队列
                if self.iteration_count < iterations:
                    retry_count = 0
                    max_retries = 5

                    while retry_count < max_retries:
                        selected_seed = self.tree.select_forest_root()
                        current_node, trajectory = self._simulate_from_seed(selected_seed)
                        direction = current_node.get_best_direction(self.exploration_weight)
                        
                        new_config = self._generate_config_with_llm(current_node, direction)
                        
                        if not self._is_duplicate_config(new_config):
                            candidate_id = f"iter_{self.iteration_count}_dir_{direction}"
                            self.iteration_count += 1
                            
                            self.task_queue.put((candidate_id, new_config))
                            self._add_config_to_seen(new_config)
                            
                            pending_tasks[candidate_id] = (current_node, direction, trajectory, new_config)
                            print(f"📤 补充任务 {candidate_id} (待完成: {iterations - completed_count})")
                            break
                        else:
                            retry_count += 1
                            self.duplicate_count += 1
                            print(f"🔁 重复配置，重试 {retry_count}/{max_retries}")
                    if retry_count == max_retries:
                        print(f"⚠️  多次重试仍重复，跳过本次补充")
            
            except Empty:
                # 队列为空，继续等待
                print(f"⏳ 等待结果... (待处理: {len(pending_tasks)})")
                continue
        
        print(f"\n✅ 搜索完成！共完成 {completed_count} 次迭代")
        self._stop_workers()
    
    def _get_root_seed_id(self, node: MCTSNode) -> str:
        """追溯到根种子"""
        current = node
        while not current.is_forest_root:
            # 找父节点
            parent_found = False
            for potential_parent in self.tree.nodes.values():  # ✅ 修正
                if current.node_id in [c.node_id for c in potential_parent.children.values()]:
                    current = potential_parent
                    parent_found = True
                    break
            if not parent_found:
                break
        return current.node_id

    # 以下方法与原版相同
    def _simulate_from_seed(self, seed_node: MCTSNode) -> Tuple[MCTSNode, List]:
        """从种子模拟（与原版相同）"""
        trajectory = []
        current_node = seed_node
        max_depth = 5
        depth = 0
        
        while depth < max_depth and current_node.children:
            direction = current_node.get_best_direction(self.exploration_weight)
            
            if direction in current_node.children:
                next_node = current_node.children[direction]
                trajectory.append((current_node, direction, next_node))
                current_node = next_node
                depth += 1
            else:
                break
        
        return current_node, trajectory
    
    def _check_memory_constraint(self, config: Dict[str, Any]) -> Tuple[bool, float, str]:
        """检查配置的内存使用情况"""
        try:
            candidate = CandidateModel(config=config)
            model = candidate.build_model()
            
            # 计算内存使用量
            memory_info = calculate_memory_usage(
                model, 
                input_size=(64, self.dataset_info['channels'], self.dataset_info['time_steps']), 
                device='cpu'  # 使用CPU计算避免GPU占用
            )
            
            memory_usage = memory_info["total_memory_MB"]
            
            # 根据量化模式调整内存使用量
            quant_mode = config.get("quant_mode", "none")
            if quant_mode in ["static", "qat", "qaft"]:
                # 量化模型通常可以压缩到原来的 1/4 左右
                compressed_memory = memory_usage / 4.0
                print(f"📦 量化压缩: {memory_usage:.2f}MB → {compressed_memory:.2f}MB (模式: {quant_mode})")
                memory_usage = compressed_memory
            
            # 检查是否超过限制
            if memory_usage <= self.max_memory:
                return True, memory_usage, "OK"
            else:
                error_msg = f"内存使用 {memory_usage:.2f}MB 超过限制 {self.max_memory}MB"
                return False, memory_usage, error_msg
        
        except Exception as e:
            print(f"⚠️ 内存计算失败: {e}")
            return False, 0, f"内存计算失败: {str(e)}"

    def _generate_config_with_llm(self, parent_node: MCTSNode, direction: str) -> Dict[str, Any]:
        """ 使用LLM生成配置 - 包含内存检查和重试机制 """
        parent_config = parent_node.candidate.config if parent_node.candidate else {}
        parent_performance = {
            "average_reward": parent_node.average_reward,
            "visit_count": parent_node.visit_count,
            "directions_explored": list(parent_node.direction_visits.keys())
        }
        global_insights = self.tree.global_insights
        
        # 🔵 重试机制
        memory_feedback = None
        llm_failed = False  # 新增：标记LLM是否失败

        for attempt in range(self.max_retry_attempts + 1):
            # ✅ 如果LLM连续2次失败，直接切换到降级策略
            try:
                # 🔵 如果LLM失败，直接使用降级配置，不再重试LLM
                if attempt >= 2 and (llm_failed or memory_feedback is not None):
                    print(f"⚠️ LLM已失败，使用降级配置 (尝试 {attempt+1}/{self.max_retry_attempts+1})")
                    # ✅ 修正：传入一个超过限制的值，触发降级策略
                    # 使用 max_memory * 2.0 表示需要强力降级到安全范围
                    new_config = self.degradation_manager.generate_degraded_config(
                        parent_config, direction, self.max_memory
                    )
                else:
                    try:
                        # 🔵 调用LLM生成配置
                        new_config = self.llm_config_generator.generate_config_with_context(
                            parent_config, direction, parent_performance, global_insights,
                            memory_feedback=memory_feedback
                        )
                        
                    except Exception as llm_e:
                        print(f"❌ LLM调用失败: {llm_e}")
                        llm_failed = True
                        # ✅ 修正：同样传入超过限制的值
                        new_config = self.degradation_manager.generate_degraded_config(
                            parent_config, direction, self.max_memory
                        )
                print(f"🔧 生成配置 (尝试 {attempt+1}/{self.max_retry_attempts+1})")

                # 🔵 检查内存约束
                memory_ok, memory_usage, memory_msg = self._check_memory_constraint(new_config)
                print(f"📊 内存使用: {memory_usage:.2f}MB (限制: {self.max_memory}MB)")

                # 🔵 检查是否重复
                if self._is_duplicate_config(new_config):
                    print(f"🔁 LLM生成重复配置 (尝试 {attempt+1}/{self.max_retry_attempts+1})")

                    # 如果是最后一次尝试，强制使用降级配置并添加随机扰动
                    if attempt == self.max_retry_attempts:
                        print("🚨 达到最大重试次数，使用强制降级配置")
                        # 使用降级配置，但指定不同的内存预算以获得不同的配置
                        import random
                        # ✅ 修正：使用当前测量的内存值，或者使用一个略高于限制的值
                        fallback_memory = memory_usage if memory_usage > 0 else self.max_memory * 1.1
                        return self.degradation_manager.generate_degraded_config(
                            parent_config, direction, fallback_memory
                        )

                    memory_feedback = f"""
                    The previous model config: {json.dumps(new_config)}
                    The generated configuration is a DUPLICATE of a previously seen model. 
                    Please generate a DIFFERENT architecture. This is attempt {attempt+1}/{self.max_retry_attempts+1}.

                    Suggestions to create a unique configuration:
                    - Change the number of stages (currently: {len(new_config.get('stages', []))})
                    - Modify the number of blocks in each stage
                    - Use different convolution types (MBConv, DWSepConv, SeSepConv, DpConv, SeDpConv)
                    - Adjust channel numbers significantly
                    - Try different expansion ratios
                    - Modify SE module settings
                    """
                    continue
                
                # 🔵 检查内存是否通过
                if memory_ok:
                    print(f"✅ 配置通过所有检查: {memory_usage:.2f}MB")
                    return new_config
                else:
                    print(f"⚠️ 配置内存超标 ({attempt+1}/{self.max_retry_attempts+1}): {memory_msg}")
                
                    # 如果是最后一次尝试，使用降级配置
                    if attempt == self.max_retry_attempts:
                        print("🚨 达到最大重试次数，使用降级配置")
                        # ✅ 调用降级管理器
                        return self.degradation_manager.generate_degraded_config(
                            parent_config, direction, memory_usage
                        )

                    # ✅ 改进后的反馈（更具体、更智能）
                    reduction_needed = memory_usage / self.max_memory
                    specific_suggestions = []

                    if reduction_needed > 1.5:
                        specific_suggestions.append("CRITICAL: Memory is 50%+ over limit. Reduce stages to 2-3 maximum")
                        specific_suggestions.append(f"Set all channel numbers to 8-16 range")
                    elif reduction_needed > 1.2:
                        specific_suggestions.append(f"Reduce channel numbers by ~{int((reduction_needed-1)*100)}%")
                        specific_suggestions.append("Remove 1-2 blocks from each stage")

                    # 添加"避免重复"的指导
                    if attempt > 1:
                        specific_suggestions.append("⚠️ IMPORTANT: Previous attempts generated duplicates. Try:")
                        specific_suggestions.append("  - Use different conv types (e.g., DpConv instead of SeDpConv)")
                        specific_suggestions.append(f"  - Use unusual channel numbers (e.g., 11, 13, 19 instead of 8, 16)")
                        specific_suggestions.append("  - Vary expansion ratios (2, 3, 5 instead of common 4, 6)")


                    # 更新反馈，要求减少内存
                    memory_feedback = f"""
                    Previous config memory: {memory_usage:.2f}MB (limit: {self.max_memory}MB)
                    Over budget by: {(reduction_needed-1)*100:.0f}%
                    
                    SPECIFIC ACTIONS REQUIRED:
                    {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(specific_suggestions))}

                    Current config:\n {json.dumps(new_config, indent=2)}
                    """
            except Exception as e:
                print(f"❌ LLM配置生成失败 (尝试 {attempt+1}): {e}")
                if attempt == self.max_retry_attempts:
                    # ✅ 调用降级管理器
                    return self.degradation_manager.generate_degraded_config(
                        parent_config, direction, self.max_memory
                    )
                
        # 最终回退
        print("🚨 所有尝试失败，使用降级配置")
        # ✅ 修正：传入超过限制的值触发降级
        return self.degradation_manager.generate_degraded_config(parent_config, direction, self.max_memory)
        

    def _backpropagate(self, trajectory: List, reward: float):
        """反向传播（与原版相同）"""
        current_reward = reward
        
        for i, (node, direction, next_node) in enumerate(reversed(trajectory)):
            node.update_direction_stats(direction, current_reward)
            
            current_q = node.direction_q_values.get(direction, 0.0)
            next_max_q = 0.0
            if next_node and next_node.direction_q_values:
                next_max_q = max(next_node.direction_q_values.values())
            
            new_q = max(current_q, next_max_q)
            visits = node.direction_visits.get(direction, 0)
            alpha = 1.0 / (visits + 1) if visits > 0 else 1.0
            updated_q = (1 - alpha) * current_q + alpha * new_q
            
            node.direction_q_values[direction] = updated_q
            node.update_reward(current_reward)
            current_reward = node.average_reward * 0.9
    
    def _is_duplicate_config(self, config: Dict[str, Any]) -> bool:
        """检查重复"""
        config_hash = self._generate_config_hash(config)
        return config_hash in self.seen_configs
    
    def _add_config_to_seen(self, config: Dict[str, Any]):
        """添加到已见集合"""
        config_hash = self._generate_config_hash(config)
        self.seen_configs.add(config_hash)
    
    def _print_search_progress(self, iteration: int):
        """打印进度"""
        stats = self.tree.get_graph_statistics()
        
        print(f"\n--- 迭代 {iteration} 进度 ---")
        print(f"最佳奖励: {self.best_reward:.4f}")
        print(f"总节点数: {stats['total_nodes']}")
        print(f"评估节点: {stats['evaluated_nodes']}")
        print(f"重复配置: {self.duplicate_count}")
        print("----------------------------\n")
    
    def get_best_model(self) -> Optional[CandidateModel]:
        """获取最佳模型"""
        return self.best_candidate
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
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


# 工厂函数
def create_parallel_sfs_search(search_space: Dict[str, Any], constraints: Dict[str, float],
                               dataset_name: str = "MMAct", num_gpus: int = 4,
                               device: str = "cuda", exploration_weight: float = 1.414) -> ParallelScatteredForestSearch:
    """创建并行SFS搜索实例"""
    return ParallelScatteredForestSearch(
        search_space=search_space,
        constraints=constraints,
        dataset_name=dataset_name,
        num_gpus=num_gpus,
        device=device,
        exploration_weight=exploration_weight
    )