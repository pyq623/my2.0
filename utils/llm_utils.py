# tinyml/utils/llm_utils.py
import yaml
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class LLMInitializer:
    """
    集中管理 LLM 初始化的工具类，确保所有模块使用统一的LLM配置
    """
    _instance = None
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # 从YAML文件加载配置（如果未提供参数）
        if not kwargs and config_path:
            config = self._load_config(config_path)
            kwargs = config['llm']  # 提取llm配置部分
        
        self.llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.7),
            base_url=kwargs["base_url"],
            api_key=kwargs["api_key"]
        )
    
    @classmethod
    def _load_config(cls, config_path: str) -> dict:
        """加载YAML配置文件"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return yaml.safe_load(path.read_text())
    
    @classmethod
    def initialize(cls, config_path: str = None, **kwargs):
        """初始化LLM单例"""
        if cls._instance is None:
            cls._instance = cls(config_path, **kwargs)
        return cls._instance
    
    @classmethod
    def get_llm(cls):
        """获取已初始化的LLM实例"""
        if cls._instance is None:
            default_config = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
            cls.initialize(config_path=default_config)
        return cls._instance.llm


def initialize_llm(llm_config: dict = None):
    """
    初始化LLM的工厂函数
    :param llm_config: 可选，如果不提供则从默认配置文件读取
    """
    if llm_config is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
        return LLMInitializer.initialize(config_path=config_path).get_llm()
    return LLMInitializer.initialize(**llm_config).get_llm()

def call_llm_with_messages(system_prompt: str, human_prompt: str, llm_instance = None) -> str:
    """
    使用 SystemMessage 和 HumanMessage 调用 LLM 的便捷函数
    
    Args:
        system_prompt: 系统提示词
        human_prompt: 用户提示词  
        llm_instance: 可选的 LLM 实例，如果为 None 则使用默认实例
    
    Returns:
        LLM 的响应文本
    """
    if llm_instance is None:
        llm_instance = initialize_llm()
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    response = llm_instance.invoke(messages)
    return response.content

# if __name__ == "__main__":
#     system_prompt = "You are a code expert."
#     human_prompt = "write down a Hanoi tower function."
#     response = call_llm_with_messages(system_prompt, human_prompt)
#     print(f"response: {response}")
