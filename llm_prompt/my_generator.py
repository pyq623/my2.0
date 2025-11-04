# models/llm_config_generator.py
import json
import re
from typing import Dict, Any, Optional
from utils.llm_utils import call_llm_with_messages
from data import get_dataset_info

class LLMConfigGenerator:
    """LLM 配置生成器 - 负责与 LLM 交互生成模型配置"""
    
    def __init__(self, search_space: Dict[str, Any], constraint: Dict[str, Any], dataset_name: str):
        self.search_space = search_space
        self.constraint = constraint
        # self.llm = initialize_llm()
        self.global_insights = []  # 存储跨分支的成功经验
        self.dataset_info = get_dataset_info(dataset_name)
        
    def generate_config_with_context(self, parent_config: Dict[str, Any], 
                                   direction: str,
                                   parent_performance: Dict[str, Any],
                                   global_insights: Dict = None,  # 新增参数
                                   memory_feedback: str = None) -> Dict[str, Any]:
        """基于上下文生成新配置"""
        
        # 构建提示词
        system_prompt = self._build_system_prompt(direction, global_insights)

        try:
            # 使用当前的内存反馈构建提示词
            human_prompt = self._build_human_prompt(
                parent_config, direction, parent_performance, memory_feedback
            )
            
            print(f"🔍 调试信息 - 系统提示词长度: {len(system_prompt)}")
            print(f"🔍 调试信息 - 用户提示词长度: {len(human_prompt)}")
            # 使用便捷函数调用 LLM
            response = call_llm_with_messages(system_prompt, human_prompt)
            
            print(f"🔍 调试信息 - LLM 响应: {response[:200]}...")  # 只打印前 200 个字符

            # 解析响应并提取 JSON 配置
            new_config = self._parse_llm_response(response)
            
            # 验证配置的有效性并自动修复
            validated_config = self._validate_and_fix_config(new_config, direction)
            if validated_config:
                # 记录成功的配置经验
                # self._update_global_insights(direction, new_config)
                return validated_config
            else:
                print("LLM 生成的配置无效，使用智能变异")
                return self._generate_base_config(direction)
                
        except Exception as e:
            print(f"LLM 配置生成失败: {e}，使用智能变异")
            return self._generate_base_config(direction)
    
    def _build_system_prompt(self, direction: str, global_insights: Dict = None) -> str:
        """构建系统提示词"""
        # 构建经验部分
        insights_text = ""

        if global_insights:
            useful_insights = []
            
            # 提取与当前方向相关的经验
            for insight_key, insight_data in global_insights.items():
                if insight_key.startswith("direction_"):
                    dir_name = insight_key.replace("direction_", "")
                    success_rate = insight_data.get('success_rate', 0)
                    avg_reward = insight_data.get('average_reward', 0)
                    
                    # 只显示成功率较高的经验
                    if success_rate > 0.6:
                        useful_insights.append(
                            f"方向 '{dir_name}': 成功率 {success_rate:.1%}, 平均奖励 {avg_reward:.3f}"
                        )
            if useful_insights:
                insights_text = f"""
                GLOBAL EXPERIENCE INSIGHTS (from successful search branches):
                {chr(10).join(f"• {insight}" for insight in useful_insights)}
                """

        system_prompt = f"""
        You are a neural network architecture design expert. Your task is to generate improved network configurations based on given constraints and insights.
        {insights_text}

        **Conv Type in the Search Space:**
            1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
            2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
            3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
            4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
            5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.
        
        **Quantization Modes (IMPORTANT):**
            - none: No quantization - standard FP32 model (baseline)
            - static: Post-training static quantization - applies INT8 quantization after training (fast but may lose accuracy)
            - qat: Quantization-Aware Training - simulates quantization effects during training to improve accuracy after quantization
            - qaft: Quantization-Aware Fine-Tuning - fine-tunes a pre-trained model with quantization awareness to recover accuracy lost during quantization
            * RECOMMENDED for best accuracy-efficiency trade-off
            * Quantization may yield dramatic degration or slight degration in accuracy depending on model architecture.

        **Important Notes:**
            - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
            - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
            - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
            - "MBConv" is only different from "DWSeqConv" when expansion > 1, otherwise they are the same block.
            - If the type of a convolution block is "SeDpConv", then the `in_channels` and `out_channels` of this convolution block must be equal. This means that: - The `out_channels` of the previous convolution block must be equal to both the `in_channels` and `out_channels` of "SeDpConv".
            - If "SeDpConv" is a block in the first stage, its `channels` should be equal to `input_channels`, otherwise an error will be reported.
            
        Please ensure the configurations adhere to the defined search space constraints.
        """
        return system_prompt
    
    def _build_human_prompt(self, parent_config: Dict[str, Any], 
                          direction: str,
                          parent_performance: Dict[str, Any],
                          memory_feedback: str = None) -> str:
        """构建用户提示词"""
        channels = self.dataset_info['channels']
        time_steps = self.dataset_info['time_steps']
        num_classes = self.dataset_info['num_classes']
        parent_config = json.dumps(parent_config, indent=2) if parent_config else "No parent configuration available"
        
        max_peak_memory = self.constraint["max_peak_memory"]
        avg_reward = parent_performance.get('average_reward', 0)
        avg_reward = f"{avg_reward:.4f}"  # 限制为小数点后4位
        visit_count = parent_performance.get('visit_count', 0)
        directions_explored = parent_performance.get('directions_explored', [])

        # 添加内存反馈到提示词
        memory_feedback_section = ""
        if memory_feedback:
            memory_feedback_section = f"""
            MEMORY FEEDBACK (Important!):
            {memory_feedback}
            
            Please generate a configuration that strictly adheres to the memory constraint.
            """

        human_prompt = """
        Parent Network Configuration:
        {parent_config}

        Parent Network Performance:
        - Average Reward: {avg_reward}
        - Visit Count: {visit_count}
        - Directions Explored: {directions_explored}

        Optimization Direction: {direction}
        (e.g., 'none' for origianl model without quantization, 'static' for static quantization, 'qat' for quantization-aware training, 'qaft' for Quantization-Aware Fine-Tuning.)

        {memory_feedback}
        
        TASK: Generate a new improved network configuration that:
        1. Uses the specified direction: {direction}
        2. Learns from the global experience insights above  
        3. Explores novel but promising architectural changes
        4. Maintains compatibility with the search space constraints, which is a must!
        5. Optimization direction is the specified quantization mode for this generation. The generated configured quantization mode must be the optimization direction.
        
        CONSTRAINTS (Search Space):
        Max memory: {max_peak_memory} MB

        SEARCH SPACE:
        {search_space}

        Please generate a new improved network configuration in valid JSON format. 
        The config you generated should have the same input channels and num classes in the example.
        For example:
        ```json
        {{
                "input_channels": {channels},  
                "num_classes": {classes},
                "quant_mode": "none",
                "stages": [
                    {{
                        "blocks": [
                            {{
                                "type": "DWSepConv",
                                "kernel_size": 3,
                                "expansion": 3,
                                "has_se": false,
                                "se_ratios": 0,
                                "skip_connection": false,
                                "stride": 1,
                                "activation": "ReLU6"
                            }}
                        ],
                        "channels": 8
                    }},
                    {{
                        "blocks": [
                            {{
                                "type": "MBConv",
                                "kernel_size": 3,
                                "expansion": 4,
                                "has_se": true,
                                "se_ratios": 0.25,
                                "skip_connection": true,
                                "stride": 2,
                                "activation": "Swish"
                            }}
                        ],
                        "channels": 16
                    }}
                ]
            }}
        ```
        
        Return ONLY the JSON configuration.""".format(
            direction = direction,
            parent_config = parent_config,
            channels=channels,
            classes=num_classes,
            avg_reward = avg_reward,
            visit_count = visit_count,
            directions_explored = directions_explored,
            max_peak_memory=max_peak_memory,
            search_space=json.dumps(self.search_space, indent=2),
            memory_feedback=memory_feedback_section  
        )
        return human_prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 响应，提取 JSON 配置"""
        try:
            # 尝试直接解析 JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果没有找到 JSON，尝试解析为代码块
                code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON 解析失败: {e}")
            raise
    
    def _validate_and_fix_config(self, config: Dict[str, Any], target_direction: str) -> Dict[str, Any]:
        """验证配置的有效性"""
        required_fields = ['input_channels', 'num_classes', 'quant_mode', 'stages']
        
        # 检查必需字段
        for field in required_fields:
            if field not in config:
                print(f"Missing required field: {field}")
                return None
        
        # 检查阶段结构
        if not isinstance(config['stages'], list):
            print("Stages must be a list")
            return None
        
        # 强制修复量化模式
        if config.get('quant_mode') != target_direction:
            print(f"Auto-fixing: quant_mode should be '{target_direction}', but got '{config.get('quant_mode')}'. Setting to '{target_direction}'")
            config['quant_mode'] = target_direction
        
        # 修复和验证每个阶段
        input_channels = config['input_channels']
        current_channels = input_channels
        
        for stage_idx, stage in enumerate(config['stages']):
            if 'channels' not in stage or 'blocks' not in stage:
                print("Invalid stage structure")
                return None
            
            if not isinstance(stage['blocks'], list) or len(stage['blocks']) == 0:
                print("Blocks must be a non-empty list")
                return None
            
            stage_channels = stage['channels']
            
            # 验证和修复每个block
            for block_idx, block in enumerate(stage['blocks']):
                if not self._validate_and_fix_block(block, stage_idx, block_idx, current_channels, stage_channels, input_channels):
                    return None
                
                # 处理 SeDpConv 的通道修复
                if block['type'] == 'SeDpConv' and '_stage_channels_fixed' in block:
                    # 应用通道修复
                    fixed_channels = block['_stage_channels_fixed']
                    stage['channels'] = fixed_channels
                    print(f"Applied SeDpConv channel fix: stage {stage_idx} channels set to {fixed_channels}")
                    # 移除临时标记
                    del block['_stage_channels_fixed']
                    # 更新当前通道数
                    current_channels = fixed_channels
                else:
                    # 更新当前通道数用于下一个block
                    if block['type'] == 'SeDpConv':
                        # SeDpConv 保持通道数不变
                        current_channels = current_channels
                    else:
                        current_channels = stage['channels']
        
        return config
    
    def _validate_and_fix_block(self, block: Dict[str, Any], stage_idx: int, block_idx: int, 
                              current_channels: int, stage_channels: int, input_channels: int) -> bool:
        """验证和修复单个 block 的配置"""
        required_block_fields = ['type', 'kernel_size', 'stride', 'expansion', 'has_se', 'se_ratio', 'skip_connection', 'activation']
        
        # 检查必需字段
        for field in required_block_fields:
            if field not in block:
                print(f"Block missing required field: {field}")
                return False
        
        # 验证卷积类型
        valid_conv_types = ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"]
        if block['type'] not in valid_conv_types:
            print(f"Invalid conv type: {block['type']}")
            return False
        
        # 处理 SeDpConv 的特殊约束
        if block['type'] == 'SeDpConv':
            # SeDpConv 要求输入输出通道相等
            if stage_idx == 0 and block_idx == 0:
                # 第一个stage的第一个block，通道数必须等于input_channels
                if stage_channels != input_channels:
                    print(f"Auto-fixing: SeDpConv in first block requires channels equal to input_channels ({input_channels}), changing from {stage_channels} to {input_channels}")
                    # 直接修改stage的channels
                    block['_stage_channels_fixed'] = input_channels  # 标记需要修复
            else:
                # 其他位置的SeDpConv，通道数必须等于当前输入通道
                if stage_channels != current_channels:
                    print(f"Auto-fixing: SeDpConv requires channels equal to input channels ({current_channels}), changing from {stage_channels} to {current_channels}")
                    # 直接修改stage的channels
                    block['_stage_channels_fixed'] = current_channels  # 标记需要修复
        
        # 验证和修复 SE 模块设置
        if block['type'] in ['SeDpConv', 'SeSepConv']:
            # 这些类型必须启用 SE
            if not block['has_se']:
                print(f"Auto-fixing: {block['type']} must have has_se=True")
                block['has_se'] = True
            
            if block['se_ratio'] <= 0:
                # 设置合理的默认 SE ratio
                block['se_ratio'] = 0.25
                print(f"Auto-fixing: {block['type']} must have se_ratio > 0, set to 0.25")
        
        # 验证 has_se 和 se_ratio 的一致性
        if block['has_se'] and block['se_ratio'] <= 0:
            print(f"Auto-fixing: has_se=True but se_ratio={block['se_ratio']}, setting se_ratio=0.25")
            block['se_ratio'] = 0.25
        elif not block['has_se'] and block['se_ratio'] > 0:
            print(f"Auto-fixing: has_se=False but se_ratio={block['se_ratio']}, setting se_ratio=0")
            block['se_ratio'] = 0
        
        # 验证和修复 skip connection
        if block['type'] in ['DpConv', 'SeDpConv', 'SeSepConv']:
            # 这些类型不支持 skip connection
            if block['skip_connection']:
                print(f"Auto-fixing: {block['type']} does not support skip_connection, setting to False")
                block['skip_connection'] = False
        
        # 验证 MBConv 和 DWSepConv 的关系
        if block['type'] == 'MBConv' and block['expansion'] == 1:
            # MBConv with expansion=1 实际上就是 DWSepConv
            print(f"Auto-fixing: MBConv with expansion=1 is equivalent to DWSepConv")
            block['expansion'] = 2
        elif block['type'] == 'DWSepConv' and block['expansion'] > 1:
            # DWSepConv with expansion>1 实际上就是 MBConv
            print(f"Auto-fixing: DWSepConv with expansion>1 is equivalent to MBConv")
            block['expansion'] = 1
        
        # 验证 expansion 范围
        if block['expansion'] < 1:
            print(f"Auto-fixing: expansion cannot be < 1, setting to 1")
            block['expansion'] = 1
        
        # 验证 kernel_size
        if block['kernel_size'] not in [1, 3, 5, 7]:
            print(f"Auto-fixing: invalid kernel_size {block['kernel_size']}, setting to 3")
            block['kernel_size'] = 3
        
        # 验证 stride (第一个block可以有stride>1，其他block应该为1)
        if block_idx > 0 and block['stride'] > 1:
            print(f"Auto-fixing: only first block in stage can have stride>1, setting stride to 1")
            block['stride'] = 1
        
        # 验证 activation
        valid_activations = ['ReLU', 'ReLU6', 'Swish', 'HSwish', 'LeakyReLU']
        if block['activation'] not in valid_activations:
            print(f"Auto-fixing: invalid activation {block['activation']}, setting to ReLU")
            block['activation'] = 'ReLU'
        
        return True
    
    def _generate_base_config(self, direction: str) -> Dict[str, Any]:
        """生成基础配置（回退方案）"""
        print(f"生成基础配置，方向: {direction}")
        
        base_config = {
            "input_channels": self.dataset_info['channels'],
            "num_classes": self.dataset_info['num_classes'],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "DWSepConv",
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratio": 0,
                            "skip_connection": False,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": 8
                },
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "stride": 2,
                            "expansion": 2,
                            "has_se": True,
                            "se_ratio": 0.25,
                            "skip_connection": True,
                            "activation": "Swish"
                        }
                    ],
                    "channels": 16
                }
            ]
        }
        
        return base_config
    
