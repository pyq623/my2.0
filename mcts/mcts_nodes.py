# models/mcts_nodes.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import uuid
from models import CandidateModel
import numpy as np

@dataclass
class MCTSNode:
    """MCTS搜索树节点"""
    node_id: str
    candidate: CandidateModel
    parent: Optional['MCTSNode'] = None
    parent_direction: Optional[str] = None  # 修改2: 记录来自父节点的方向
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    total_reward: float = 0.0
    directions: List[str] = field(default_factory=list)  # 量化模式方向
    direction_q_values: Dict[str, float] = field(default_factory=dict)  # 方向Q值
    direction_visits: Dict[str, int] = field(default_factory=dict)  # 方向访问次数

    # 修改3: 添加图结构相关的字段
    graph_children: Set[str] = field(default_factory=set)  # 图中所有子节点的ID
    is_forest_root: bool = False  # 是否是森林中的根节点
    
    def __post_init__(self):
        """初始化方向相关的字典"""
        if not self.directions:
            self.directions = ["none", "static", "qat"]
        
        for direction in self.directions:
            self.direction_q_values[direction] = 0.0
            self.direction_visits[direction] = 0
    
    @property
    def average_reward(self) -> float:
        """计算平均奖励"""
        return self.total_reward / max(1, self.visit_count)
    
    def add_child(self, direction: str, child_node: 'MCTSNode'):
        """添加子节点"""
        self.children[direction] = child_node
        self.graph_children.add(child_node.node_id)  # 修改4: 维护图结构关系
        child_node.parent = self
        child_node.parent_direction = direction  # 记录方向信息
    
    def update_reward(self, reward: float):
        """更新节点奖励"""
        self.visit_count += 1
        self.total_reward += reward
    
    def update_direction_stats(self, direction: str, reward: float):
        """更新方向统计信息"""
        if direction in self.direction_visits:
            self.direction_visits[direction] += 1
            # Q值更新公式
            current_q = self.direction_q_values[direction]
            visits = self.direction_visits[direction]
            self.direction_q_values[direction] = current_q + (reward - current_q) / visits
    
    def get_best_direction(self, exploration_weight: float = 1.0) -> str:
        """使用UCT公式选择最佳方向 - 修正公式
        在当前节点选择下一步要探索的最佳量化方向"""
        best_direction = None
        best_score = -float('inf')
        total_visits = sum(self.direction_visits.values())
        
        for direction in self.directions:
            if self.direction_visits[direction] == 0:
                # 未探索的方向优先选择
                return direction
            
            # 修改5: 修正UCT公式，移除多余的系数2
            exploitation = self.direction_q_values[direction]
            # exploration = exploration_weight * (2 * np.log(total_visits) / self.direction_visits[direction]) ** 0.5
            exploration = exploration_weight * np.sqrt(
                np.log(total_visits) / self.direction_visits[direction]
            )
            uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_direction = direction
        
        return best_direction
    
    def get_graph_info(self) -> Dict[str, Any]:
        """获取节点的图结构信息"""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "parent_direction": self.parent_direction,
            "children_count": len(self.graph_children),
            "children_ids": list(self.graph_children),
            "is_forest_root": self.is_forest_root
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - 增强版，包含图结构信息"""
        base_info = {
            "node_id": self.node_id,
            "candidate_config": self.candidate.config if self.candidate else None,
            "metrics": self.candidate.metrics if self.candidate else {},
            "visit_count": self.visit_count,
            "average_reward": self.average_reward,
            "children_directions": list(self.children.keys()),
            "direction_stats": {
                dir: {
                    "q_value": self.direction_q_values[dir],
                    "visits": self.direction_visits[dir]
                } for dir in self.directions
            },
            "graph_info": self.get_graph_info()  # 修改6: 添加图结构信息
        }
        return base_info


class MCTSTree:
    """MCTS图结构 - 替换原来的MCTSTree，支持真正的图结构"""
    
    def __init__(self, exploration_weight: float = 1.414):
        # 修改7: 使用图结构的数据组织方式
        self.nodes: Dict[str, MCTSNode] = {}
        self.edges: Dict[str, Dict[str, MCTSNode]] = {}  # parent_id -> {direction: child_node}
        self.parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.forest_roots: List[MCTSNode] = []  # 多棵树的根节点（Foresting）
        self.root: Optional[MCTSNode] = None
        self.exploration_weight = exploration_weight

        # 修改8: 全局记忆，用于 Scouting
        self.global_insights: Dict[str, Dict[str, Any]] = {}
    
    def add_node(self, node: MCTSNode, parent: Optional[MCTSNode] = None, 
                 direction: Optional[str] = None, is_forest_root: bool = False):
        """添加节点到树中"""
        self.nodes[node.node_id] = node
        
        if is_forest_root:
            node.is_forest_root = True
            self.forest_roots.append(node)

        if parent:
            # 建立父子关系
            if parent.node_id not in self.edges:
                self.edges[parent.node_id] = {}
            self.edges[parent.node_id][direction] = node
            self.parent_map[node.node_id] = parent.node_id
            
            # 同时更新节点的直接关系
            parent.add_child(direction, node)
    
    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """根据ID获取节点"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[MCTSNode]:
        """获取指定节点的所有子节点"""
        if node_id not in self.edges:
            return []
        return list(self.edges[node_id].values())
    
    def get_parent(self, node_id: str) -> Optional[MCTSNode]:
        """获取父节点"""
        parent_id = self.parent_map.get(node_id)
        return self.nodes.get(parent_id) if parent_id else None
    
    def get_forest_roots(self) -> List[MCTSNode]:
        """获取所有森林根节点"""
        return self.forest_roots
    
    def select_forest_root(self) -> MCTSNode:
        """使用UCT选择森林中的根节点进行扩展
        在多棵搜索树（森林）中选择要扩展的根节点"""
        if not self.forest_roots:
            # 如果没有根节点，创建一个
            root = MCTSNode()
            root.is_forest_root = True
            self.add_node(root, is_forest_root=True)
            return root
        
        # 使用UCT公式选择根节点
        best_root = None
        best_score = -float('inf')
        
        for root in self.forest_roots:
            if root.visit_count == 0:
                return root
            
            # 使用平均奖励作为利用项
            exploitation = root.average_reward
            # 探索项：鼓励访问次数少的根节点
            total_visits = sum(r.visit_count for r in self.forest_roots)
            exploration = self.exploration_weight * np.sqrt(
                np.log(total_visits) / root.visit_count
            )
            uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_root = root
        
        return best_root or self.forest_roots[0]

    def scattering(self, node: MCTSNode) -> List[str]:
        """Scattering: 生成多样化的量化方向"""
        # 基于全局insights调整方向优先级
        base_directions = ["none", "static", "qat"]
        
        # 如果有全局经验，调整方向顺序
        scored_directions = []
        for direction in base_directions:
            insight_key = f"direction_{direction}"
            if insight_key in self.global_insights:
                insight = self.global_insights[insight_key]
                score = insight.get('success_rate', 0.5)
            else:
                score = 0.5
            scored_directions.append((direction, score))
        
        # 按成功率排序
        scored_directions.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_directions]
    
    def scouting(self, parent: MCTSNode, direction: str, child: MCTSNode, reward: float, feedback: Dict = None):
        """Scouting: 跨分支共享经验"""
        insight_key = f"direction_{direction}"
        
        if insight_key not in self.global_insights:
            self.global_insights[insight_key] = {
                'total_reward': 0.0,
                'visit_count': 0,
                'success_count': 0,
                'recent_rewards': [],  # 记录最近奖励用于自适应阈值
                'successful_configs': []  # 新增：记录成功的配置特征
            }
        
        insight = self.global_insights[insight_key]
        insight['total_reward'] += reward
        insight['visit_count'] += 1
        insight['recent_rewards'].append(reward)

        # 动态阈值：如果 reward 高于历史平均值的某个比例就算成功
        if insight['visit_count'] > 1:
            # 这个动态阈值只是平均值，似乎过于简单？
            avg_reward = insight['total_reward'] / insight['visit_count']
            # 使用动态阈值，比如高于平均值的就算成功
            if reward > avg_reward * 0.9:  # 可调整的系数
                insight['success_count'] += 1
                # 记录成功的配置特征
                if feedback and 'child_config' in feedback:
                    config_insight = self._extract_config_insight(feedback['parent_config'], feedback['child_config'], reward)
                    insight['successful_configs'].append(config_insight)
                    # 限制配置经验数量
                    if len(insight['successful_configs']) > 5:
                        insight['successful_configs'].pop(0)
        else:
            # 第一次访问时使用固定阈值
            if reward > 0.5:
                insight['success_count'] += 1

        # 限制最近奖励列表长度
        if len(insight['recent_rewards']) > 10:
            insight['recent_rewards'].pop(0)
        
        # 计算成功率
        insight['success_rate'] = insight['success_count'] / insight['visit_count']
        insight['average_reward'] = insight['total_reward'] / insight['visit_count']
    
    def _extract_config_insight(self, parent_config: Dict[str, Any], child_config: Dict[str, Any], reward: float) -> Dict[str, Any]:
        """提取父子配置的关键差异"""
        parent_stages = parent_config.get('stages', [])
        child_stages = child_config.get('stages', [])

        diff_analysis = {
            'reward': reward,
            'improvement_score': reward - (parent_config.get('metrics', {}).get('accuracy', 0) if parent_config.get('metrics') else 0),
            'major_changes': [],
            'detailed_differences': {},
            'change_summary': 'minor_changes'  # 默认
        }
        # 1. 首先检查stage数量差异
        has_stage_count_change = False
        if len(parent_stages) != len(child_stages):
            diff_analysis['major_changes'].append(f"stage_count_changed: {len(parent_stages)} -> {len(child_stages)}")
            has_stage_count_change = True
        
        # 2. 检查每个stage的block数量差异
        block_count_changes = []
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            
            if len(p_blocks) != len(c_blocks):
                block_count_changes.append(f"stage_{i}_blocks: {len(p_blocks)} -> {len(c_blocks)}")
        
        has_block_count_change = len(block_count_changes) > 0
        if block_count_changes:
            diff_analysis['major_changes'].extend(block_count_changes)
            # block数量变化也是主要变化
        
        # 3. 如果 stage 和 block 数量都相同，检查conv type差异
        conv_type_changes = []
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            
            for j, (p_block, c_block) in enumerate(zip(p_blocks, c_blocks)):
                p_conv_type = p_block.get('type', '')
                c_conv_type = c_block.get('type', '')
                
                if p_conv_type != c_conv_type:
                    conv_type_changes.append(f"stage_{i}_block_{j}_conv_type: {p_conv_type} -> {c_conv_type}")
        
        has_conv_type_change = len(conv_type_changes) > 0
        if conv_type_changes:
            diff_analysis['major_changes'].extend(conv_type_changes)
        
        # 如果有任意一个主要变化，就返回
        if diff_analysis['major_changes']:
            # 根据变化类型设置更精确的summary - 允许多个类型共存
            change_types = []
            if has_stage_count_change:
                change_types.append('structural')
            if has_block_count_change:
                change_types.append('block_count')
            if has_conv_type_change:
                change_types.append('conv_type')
            
            # 构建复合的summary
            if change_types:
                if len(change_types) == 1:
                    diff_analysis['change_summary'] = f'major_{change_types[0]}_change'
                else:
                    diff_analysis['change_summary'] = f'major_combined_change_{"_".join(change_types)}'
            else:
                diff_analysis['change_summary'] = 'major_changes'
            
            return diff_analysis
        
        # 4. 如果以上都相同，详细分析其他参数的差异
        detailed_diffs = {}
        has_significant_changes = False
        
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            stage_diffs = {}
            
            # 检查 channels 变化
            p_channels = p_stage.get('channels', 0)
            c_channels = c_stage.get('channels', 0)
            if p_channels != c_channels:
                stage_diffs['channels'] = f"{p_channels} -> {c_channels}"
                if abs(p_channels - c_channels) > 6:  # channels 变化较大
                    has_significant_changes = True
            
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            block_diffs = []
            
            for j, (p_block, c_block) in enumerate(zip(p_blocks, c_blocks)):
                block_diff = {}
                
                # 检查kernel_size
                p_kernel = p_block.get('kernel_size', 3)
                c_kernel = c_block.get('kernel_size', 3)
                if p_kernel != c_kernel:
                    block_diff['kernel_size'] = f"{p_kernel} -> {c_kernel}"
                    if abs(p_kernel - c_kernel) > 1:  # kernel变化较大
                        has_significant_changes = True
                
                # 检查stride
                p_stride = p_block.get('stride', 1)
                c_stride = c_block.get('stride', 1)
                if p_stride != c_stride:
                    block_diff['stride'] = f"{p_stride} -> {c_stride}"
                    has_significant_changes = True  # stride变化总是重要的
                
                # 检查expansion
                p_expansion = p_block.get('expansion', 1)
                c_expansion = c_block.get('expansion', 1)
                if p_expansion != c_expansion:
                    block_diff['expansion'] = f"{p_expansion} -> {c_expansion}"
                    if abs(p_expansion - c_expansion) > 1:  # expansion变化较大
                        has_significant_changes = True
                
                # 检查SE模块
                p_has_se = p_block.get('has_se', False)
                c_has_se = c_block.get('has_se', False)
                if p_has_se != c_has_se:
                    block_diff['has_se'] = f"{p_has_se} -> {c_has_se}"
                    has_significant_changes = True
                
                # 检查activation
                p_activation = p_block.get('activation', '')
                c_activation = c_block.get('activation', '')
                if p_activation != c_activation:
                    block_diff['activation'] = f"{p_activation} -> {c_activation}"
                
                if block_diff:
                    block_diffs.append({f'block_{j}': block_diff})
            
            if block_diffs:
                stage_diffs['blocks'] = block_diffs

            if stage_diffs:
                # 这个里面已经包括了stage['channels']的变化和stage['blocks']的变化
                detailed_diffs[f'stage_{i}'] = stage_diffs
        
        diff_analysis['detailed_differences'] = detailed_diffs

        # 根据是否有显著变化更新summary
        if has_significant_changes:
            diff_analysis['change_summary'] = 'significant_parameter_changes'
        elif detailed_diffs:
            diff_analysis['change_summary'] = 'minor_parameter_changes'
        else:
            diff_analysis['change_summary'] = 'no_changes'
        
        return diff_analysis

    def get_best_candidate(self) -> Optional[CandidateModel]:
        """获取图中最佳候选模型"""
        if not self.nodes:
            return None
        
        # 从所有已评估的节点中选择最佳节点
        evaluated_nodes = [node for node in self.nodes.values() 
                         if node.candidate and node.visit_count > 0]
        
        if not evaluated_nodes:
            return None
        
        best_node = max(evaluated_nodes, key=lambda node: node.average_reward)
        return best_node.candidate
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图结构统计信息"""
        total_nodes = len(self.nodes)
        total_edges = sum(len(children) for children in self.edges.values())
        forest_count = len(self.forest_roots)
        
        evaluated_nodes = [node for node in self.nodes.values() if node.visit_count > 0]
        avg_reward = np.mean([node.average_reward for node in evaluated_nodes]) if evaluated_nodes else 0
        best_reward = max([node.average_reward for node in evaluated_nodes]) if evaluated_nodes else 0
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "forest_count": forest_count,
            "evaluated_nodes": len(evaluated_nodes),
            "average_reward": avg_reward,
            "best_reward": best_reward,
            "global_insights_count": len(self.global_insights)
        }
    
    def save_graph_info(self, filepath: str):
        """保存图结构信息到文件"""
        # 1. 计算每个节点的树深度和所属种子
        node_depths = {}
        node_root_seeds = {}

        def compute_depth_and_seed(node_id: str, current_depth: int = 0, root_seed: str = None):
            """递归计算深度和所属种子"""
            if node_id in node_depths:
                return
            
            node = self.nodes.get(node_id)
            if not node:
                return
            
            # 如果是森林根节点，它就是自己的根种子
            if node.is_forest_root:
                root_seed = node_id
                current_depth = 0
            
            node_depths[node_id] = current_depth
            node_root_seeds[node_id] = root_seed
            
            # 递归处理子节点
            for direction, child in node.children.items():
                compute_depth_and_seed(child.node_id, current_depth + 1, root_seed)
        
        # 为所有森林根节点计算深度
        for root in self.forest_roots:  # ✅ 修正
            compute_depth_and_seed(root.node_id)  # ✅ 修正
        
        # 2. 构建节点信息（增强版）
        nodes_info = {}
        for node_id, node in self.nodes.items():  # ✅ 修正
            # 找到父节点信息
            parent_id = None
            parent_direction = None
            for potential_parent_id, potential_parent in self.nodes.items():  # ✅ 修正
                for direction, child in potential_parent.children.items():
                    if child.node_id == node_id:
                        parent_id = potential_parent_id
                        parent_direction = direction
                        break
            
            # ✅ 增强的节点信息
            nodes_info[node_id] = {
                "node_id": node_id,
                "candidate_config": node.candidate.config if node.candidate else None,
                "metrics": {
                    "accuracy": node.candidate.metrics.get('accuracy', 0.0) if node.candidate else 0.0,
                    "latency": node.candidate.metrics.get('latency', 0.0) if node.candidate else 0.0,
                    "peak_memory": node.candidate.metrics.get('peak_memory', 0.0) if node.candidate else 0.0
                },
                "visit_count": node.visit_count,
                "average_reward": node.average_reward,
                "children_directions": list(node.children.keys()),
                "direction_stats": {
                    direction: {
                        "q_value": node.direction_q_values.get(direction, 0.0),
                        "visits": node.direction_visits.get(direction, 0)
                    }
                    for direction in node.directions
                },
                # ✅ 新增：树结构元信息
                "tree_metadata": {
                    "depth": node_depths.get(node_id, 0),
                    "root_seed_id": node_root_seeds.get(node_id, None),
                    "iteration": getattr(node, 'iteration', None),  # 需要在创建节点时设置
                    "is_forest_root": node.is_forest_root
                },
                # ✅ 增强：图结构信息
                "graph_info": {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "parent_direction": parent_direction,
                    "children_count": len(node.children),
                    "children_ids": [child.node_id for child in node.children.values()],
                    "is_forest_root": node.is_forest_root
                }
            }

        import json
        # graph_data = {
        #     "statistics": self.get_graph_statistics(),
        #     "forest_roots": [root.node_id for root in self.forest_roots],
        #     "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        #     "global_insights": self.global_insights,
        # }
        # 3. 构建完整的图结构数据
        graph_data = {
            "statistics": self.get_graph_statistics(),
            "forest_roots": self.forest_roots,
            "nodes": nodes_info,
            # ✅ 新增：全局洞察
            "global_insights": self.global_insights,
            # ✅ 新增：用于LLaMA3训练的轨迹数据
            "training_trajectories": self._extract_training_trajectories(nodes_info)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _extract_training_trajectories(self, nodes_info: dict) -> list:
        """提取用于LLaMA3训练的轨迹数据"""
        trajectories = []
        
        for node_id, node_data in nodes_info.items():
            # 跳过森林根节点（没有父节点）
            if node_data['graph_info']['parent_id'] is None:
                continue
            
            parent_id = node_data['graph_info']['parent_id']
            parent_direction = node_data['graph_info']['parent_direction']
            
            if parent_id in nodes_info:
                parent_data = nodes_info[parent_id]
                
                # ✅ 构建训练样本：parent_config + direction → child_config + reward
                trajectory = {
                    "trajectory_id": f"{parent_id}_to_{node_id}",
                    "parent": {
                        "node_id": parent_id,
                        "config": parent_data['candidate_config'],
                        "reward": parent_data['average_reward'],
                        "visit_count": parent_data['visit_count']
                    },
                    "action": {
                        "direction": parent_direction,
                        "q_value": parent_data['direction_stats'].get(parent_direction, {}).get('q_value', 0.0)
                    },
                    "child": {
                        "node_id": node_id,
                        "config": node_data['candidate_config'],
                        "reward": node_data['average_reward'],
                        "metrics": node_data['metrics']
                    },
                    "context": {
                        "depth": node_data['tree_metadata']['depth'],
                        "root_seed_id": node_data['tree_metadata']['root_seed_id'],
                        "iteration": node_data['tree_metadata']['iteration']
                    }
                }
                
                trajectories.append(trajectory)
        
        return trajectories