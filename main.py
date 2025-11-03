# main.py
import os
import json
import argparse
from datetime import datetime
from mcts import ScatteredForestSearch, create_sfs_search
from configs import get_search_space

def setup_search_space(dataset_name: str = "MMAct") -> dict:
    """设置搜索空间"""
    info = get_search_space()
    search_space = info['search_space']
    constraints = info['constraints']
    return search_space, constraints

def create_output_directory(experiment_name: str) -> str:
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{experiment_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_search_results(search: ScatteredForestSearch, output_dir: str, args: dict):
    """保存搜索结果"""
    # 保存最佳模型配置
    best_model = search.get_best_model()
    if best_model:
        best_config_path = os.path.join(output_dir, "best_model_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_model.config, f, indent=2, ensure_ascii=False)
        print(f"✅ 最佳模型配置已保存: {best_config_path}")
    
    # 保存搜索统计信息
    stats = search.get_search_statistics()
    stats_path = os.path.join(output_dir, "search_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 搜索统计已保存: {stats_path}")
    
    # 保存搜索状态
    state_path = os.path.join(output_dir, "search_state.json")
    search.save_search_state(state_path)
    print(f"✅ 搜索状态已保存: {state_path}")
    
    # 保存实验参数
    args_path = os.path.join(output_dir, "experiment_args.json")
    with open(args_path, 'w') as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    print(f"✅ 实验参数已保存: {args_path}")

def print_final_results(search: ScatteredForestSearch):
    """打印最终结果"""
    print("\n" + "="*60)
    print("🎉 搜索完成! 最终结果:")
    print("="*60)
    
    best_model = search.get_best_model()
    stats = search.get_search_statistics()
    
    if best_model:
        print(f"🏆 最佳模型奖励: {stats['best_reward']:.4f}")
        print(f"📊 总迭代次数: {stats['iterations']}")
        print(f"🌳 搜索树统计:")
        print(f"   - 总节点数: {stats['tree_statistics']['total_nodes']}")
        print(f"   - 评估节点: {stats['tree_statistics']['evaluated_nodes']}")
        print(f"   - 森林大小: {stats['tree_statistics']['forest_count']}")
        print(f"   - 平均奖励: {stats['tree_statistics']['average_reward']:.4f}")
        
        print("\n🔍 全局经验洞察:")
        for direction, insight in stats['global_insights'].items():
            if direction.startswith('direction_'):
                success_rate = insight.get('success_rate', 0)
                avg_reward = insight.get('average_reward', 0)
                print(f"   - {direction}: 成功率={success_rate:.3f}, 平均奖励={avg_reward:.3f}")
    
    print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="散射森林搜索算法")
    parser.add_argument("--dataset", type=str, default="MMAct", 
                       choices=["MMAct", "Mhealth", "Wharf", "UTD-MHAD", "USCHAD"], 
                       help="数据集名称")
    parser.add_argument("--iterations", type=int, default=50, 
                       help="搜索迭代次数")
    parser.add_argument("--num_seeds", type=int, default=2,   # 以后设置为5
                       help="初始种子数量")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], 
                       help="计算设备")
    parser.add_argument("--exploration_weight", type=float, default=1.414, 
                       help="探索权重")
    parser.add_argument("--experiment_name", type=str, default="sfs", 
                       help="实验名称")
    
    args = parser.parse_args()
    
    print("🚀 开始散射森林搜索...")
    print(f"📋 实验配置:")
    print(f"   - 数据集: {args.dataset}")
    print(f"   - 迭代次数: {args.iterations}")
    print(f"   - 初始种子: {args.num_seeds}")
    print(f"   - 设备: {args.device}")
    print(f"   - 探索权重: {args.exploration_weight}")
    print(f"   - 实验名称: {args.experiment_name}")
    
    # 设置搜索空间和约束
    search_space, constraints = setup_search_space(args.dataset)

    
    # 创建输出目录
    output_dir = create_output_directory(args.experiment_name)
    print(f"📁 输出目录: {output_dir}")
    
    try:
        # 创建搜索实例
        search = create_sfs_search(
            search_space=search_space,
            constraints=constraints,
            device=args.device,
            exploration_weight=args.exploration_weight,
            dataset_name=args.dataset
        )
        
        # 初始化森林
        print("\n🌱 初始化森林...")
        search.initialize_forest(num_seeds=args.num_seeds)
        
        # 执行搜索
        print(f"\n🔍 开始搜索 ({args.iterations} 次迭代)...")
        search.search(
            iterations=args.iterations,
            exploration_weight=args.exploration_weight,
            dataset_names=[args.dataset]
        )
        
        # 保存结果
        print("\n💾 保存搜索结果...")
        save_search_results(search, output_dir, vars(args))
        
        # 打印最终结果
        print_final_results(search)
        
        print(f"\n✅ 实验完成! 结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()