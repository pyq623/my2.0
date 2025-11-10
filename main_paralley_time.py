import os
import json
import argparse
import time
from datetime import datetime, timedelta
from configs import get_search_space

# 设置时区为中国北京时区
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

# 导入新的并行搜索类
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mcts import create_parallel_sfs_search

class Timer:
    """计时器类"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """退出 with 语句时自动停止"""
        self.stop()
    
    def get_elapsed_str(self) -> str:
        """获取格式化的耗时字符串"""
        return self.format_time(self.elapsed)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}分钟 ({seconds:.1f}秒)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f}小时 ({minutes:.1f}分钟)"

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

def save_search_results(search, output_dir: str, args: dict, timing_stats: dict):
    """保存搜索结果"""
    # 保存最佳模型配置
    best_model = search.get_best_model()
    if best_model:
        # ✅ 从树中查找对应节点以获取深度信息
        best_node = None
        for node_id, node in search.tree.nodes.items():
            if node.candidate and node.candidate.candidate_id == best_model.candidate_id:
                best_node = node
                break
        
        # ✅ 计算深度（如果找到节点）
        depth = 0
        if best_node:
            current = best_node
            while current.parent and not current.parent.is_forest_root:
                depth += 1
                current = current.parent
            if current.parent:  # 如果有父节点且父节点是根，深度+1
                depth += 1
        
        # ✅ 收集完整的元信息
        best_model_data = {
            "config": best_model.config,
            "performance": {
                "reward": search.best_reward,
                "accuracy": best_model.metrics.get('accuracy', 0.0),
                "latency": best_model.metrics.get('latency', 0.0),
                "peak_memory": best_model.metrics.get('peak_memory', 0.0),
                "gpu_id": best_model.metrics.get('gpu_id', -1)
            },
            "metadata": {
                "candidate_id": best_model.candidate_id if hasattr(best_model, 'candidate_id') else 'unknown',
                "iteration_discovered": getattr(best_model, 'iteration', 'unknown'),
                "parent_seed_id": getattr(best_model, 'root_seed_id', 'unknown'),
                "depth_from_seed": depth,
                "parent_node_id": getattr(best_model, 'parent_id', None),
                "parent_direction": getattr(best_model, 'parent_direction', None)
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        best_config_path = os.path.join(output_dir, "best_model_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_model_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 最佳模型配置已保存: {best_config_path}")
    
    # 保存搜索统计信息（包含计时信息）
    stats = search.get_search_statistics()
    stats['timing'] = timing_stats  # 添加计时统计
    
    stats_path = os.path.join(output_dir, "search_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 搜索统计已保存: {stats_path}")
    
    # 保存搜索状态
    state_path = os.path.join(output_dir, "search_state.json")
    search.save_search_state(state_path)
    print(f"✅ 搜索状态已保存: {state_path}")
    
    # 保存实验参数（包含计时信息）
    experiment_data = {
        'args': args,
        'timing': timing_stats
    }
    args_path = os.path.join(output_dir, "experiment_args.json")
    with open(args_path, 'w') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 实验参数已保存: {args_path}")
    
    # 单独保存计时报告
    timing_report_path = os.path.join(output_dir, "timing_report.txt")
    with open(timing_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("⏱️  运行时间统计报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"🕐 总运行时间: {Timer.format_time(timing_stats['total_time'])}\n\n")
        
        f.write("📊 各阶段耗时:\n")
        f.write(f"  1. 初始化阶段: {Timer.format_time(timing_stats['initialization_time'])}\n")
        f.write(f"  2. 森林初始化: {Timer.format_time(timing_stats['forest_init_time'])}\n")
        f.write(f"  3. 搜索执行:   {Timer.format_time(timing_stats['search_time'])}\n")
        f.write(f"  4. 结果保存:   {Timer.format_time(timing_stats['save_time'])}\n\n")
        
        f.write("📈 搜索效率:\n")
        f.write(f"  - 总迭代次数: {timing_stats['total_iterations']}\n")
        f.write(f"  - 平均每次迭代: {Timer.format_time(timing_stats['avg_iteration_time'])}\n")
        
        if timing_stats['total_iterations'] > 0:
            throughput = 3600 / timing_stats['avg_iteration_time']  # 每小时迭代数
            f.write(f"  - 吞吐量: {throughput:.2f} 次迭代/小时\n")
        
        f.write(f"  - GPU数量: {timing_stats['num_gpus']}\n")
        f.write(f"  - GPU利用率估算: {timing_stats.get('gpu_utilization_estimate', 'N/A')}\n\n")
        
        f.write("⏰ 时间戳:\n")
        f.write(f"  - 开始时间: {timing_stats['start_time']}\n")
        f.write(f"  - 结束时间: {timing_stats['end_time']}\n")
        
    print(f"✅ 计时报告已保存: {timing_report_path}")

def print_timing_summary(timing_stats: dict):
    """打印计时摘要"""
    print("\n" + "="*60)
    print("⏱️  运行时间统计")
    print("="*60)
    
    print(f"\n🕐 总运行时间: {Timer.format_time(timing_stats['total_time'])}")
    
    print(f"\n📊 各阶段耗时:")
    # ✅ 添加安全检查
    if 'initialization_time' in timing_stats:
        print(f"  1️⃣  初始化阶段: {Timer.format_time(timing_stats['initialization_time'])} "
              f"({timing_stats['initialization_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'forest_init_time' in timing_stats:
        print(f"  2️⃣  森林初始化: {Timer.format_time(timing_stats['forest_init_time'])} "
              f"({timing_stats['forest_init_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'search_time' in timing_stats:
        print(f"  3️⃣  搜索执行:   {Timer.format_time(timing_stats['search_time'])} "
              f"({timing_stats['search_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'save_time' in timing_stats:
        print(f"  4️⃣  结果保存:   {Timer.format_time(timing_stats['save_time'])} "
              f"({timing_stats['save_time']/timing_stats['total_time']*100:.1f}%)")
    
    print(f"\n📈 搜索效率:")
    print(f"  - 总迭代次数: {timing_stats['total_iterations']}")
    print(f"  - 平均每次迭代: {Timer.format_time(timing_stats['avg_iteration_time'])}")
    
    if timing_stats['total_iterations'] > 0:
        throughput = 3600 / timing_stats['avg_iteration_time']
        print(f"  - 吞吐量: {throughput:.2f} 次迭代/小时")
    
    print(f"  - GPU数量: {timing_stats['num_gpus']}")
    
    # 估算GPU利用率
    if timing_stats['num_gpus'] > 0:
        ideal_time = timing_stats['search_time'] / timing_stats['num_gpus']
        actual_avg_time = timing_stats['avg_iteration_time']
        utilization = (ideal_time / actual_avg_time) * 100 if actual_avg_time > 0 else 0
        print(f"  - GPU利用率估算: {utilization:.1f}%")
        timing_stats['gpu_utilization_estimate'] = f"{utilization:.1f}%"
    
    print(f"\n⏰ 时间范围:")
    print(f"  开始: {timing_stats['start_time']}")
    print(f"  结束: {timing_stats['end_time']}")
    
    print("="*60)

def print_final_results(search, timing_stats: dict):
    """打印最终结果"""
    print("\n" + "="*60)
    print("🎉 搜索完成! 最终结果:")
    print("="*60)
    
    best_model = search.get_best_model()
    stats = search.get_search_statistics()
    
    if best_model:
        print(f"\n🏆 最佳模型:")
        print(f"   - 奖励: {stats['best_reward']:.4f}")
        print(f"   - 配置ID: {best_model.candidate_id if hasattr(best_model, 'candidate_id') else 'N/A'}")
        
        print(f"\n📊 搜索统计:")
        print(f"   - 总迭代次数: {stats['iterations']}")
        print(f"   - 唯一配置: {stats['unique_configs']}")
        print(f"   - 重复次数: {stats['duplicate_count']}")
        print(f"   - 重复率: {stats['duplicate_count']/(stats['iterations']+stats['duplicate_count'])*100:.1f}%")
        
        print(f"\n🌳 搜索树统计:")
        print(f"   - 总节点数: {stats['tree_statistics']['total_nodes']}")
        print(f"   - 评估节点: {stats['tree_statistics']['evaluated_nodes']}")
        print(f"   - 森林大小: {stats['tree_statistics']['forest_count']}")
        print(f"   - 平均奖励: {stats['tree_statistics']['average_reward']:.4f}")
        print(f"   - 最佳奖励: {stats['tree_statistics']['best_reward']:.4f}")
        
        print(f"\n🔍 全局经验洞察:")
        insights = stats['global_insights']
        if insights:
            for direction, insight in insights.items():
                if direction.startswith('direction_'):
                    direction_name = direction.replace('direction_', '')
                    success_rate = insight.get('success_rate', 0)
                    avg_reward = insight.get('average_reward', 0)
                    visit_count = insight.get('visit_count', 0)
                    print(f"   - {direction_name:6s}: 访问={visit_count:3d}, "
                          f"成功率={success_rate:.1%}, 平均奖励={avg_reward:.3f}")
        else:
            print("   (暂无洞察数据)")
    
    # 打印计时摘要
    print_timing_summary(timing_stats)
    
    print("\n" + "="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="并行散射森林搜索（队列版本）")
    parser.add_argument("--dataset", type=str, default="MMAct", 
                       choices=["MMAct", "Mhealth", "Wharf", "UTD-MHAD", "USCHAD"], 
                       help="数据集名称")
    parser.add_argument("--iterations", type=int, default=50, 
                       help="搜索迭代次数")
    parser.add_argument("--num_seeds", type=int, default=4,
                       help="初始种子数量")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="使用的GPU数量")
    parser.add_argument("--train_epochs", type=int, default=100,
                       help="每个模型的训练轮数")
    parser.add_argument("--exploration_weight", type=float, default=1.414, 
                       help="探索权重")
    parser.add_argument("--experiment_name", type=str, default="sfs_queue", 
                       help="实验名称")
    
    args = parser.parse_args()
    
    # 记录开始时间
    experiment_start_time = time.time()
    start_datetime = datetime.now()
    
    print("🚀 开始并行散射森林搜索（队列版本）...")
    print(f"⏰ 开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📋 实验配置:")
    print(f"   - 数据集: {args.dataset}")
    print(f"   - 迭代次数: {args.iterations}")
    print(f"   - 初始种子: {args.num_seeds}")
    print(f"   - GPU数量: {args.num_gpus}")
    print(f"   - 训练轮数: {args.train_epochs}")
    print(f"   - 探索权重: {args.exploration_weight}")
    print(f"   - 实验名称: {args.experiment_name}")
    
    # 初始化计时统计
    timing_stats = {
        'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'num_gpus': args.num_gpus,
        'total_iterations': args.iterations,
    }
    
    # 设置搜索空间和约束
    print("\n⏱️  [1/4] 初始化阶段...")
    with Timer("initialization") as init_timer:
        search_space, constraints = setup_search_space(args.dataset)
        output_dir = create_output_directory(args.experiment_name)
        print(f"📁 输出目录: {output_dir}")
    
    timing_stats['initialization_time'] = init_timer.elapsed
    print(f"   ✅ 初始化完成，耗时: {init_timer.get_elapsed_str()}")
    
    try:
        # 创建并行搜索实例
        search = create_parallel_sfs_search(
            search_space=search_space,
            constraints=constraints,
            dataset_name=args.dataset,
            num_gpus=args.num_gpus,
            exploration_weight=args.exploration_weight
        )
        
        # 初始化森林（会启动工作进程）
        print(f"\n⏱️  [2/4] 森林初始化阶段 ({args.num_seeds} 个种子)...")
        forest_init_start = time.time()
        with Timer("forest_init") as forest_timer:
            search.initialize_forest(num_seeds=args.num_seeds)
        
        timing_stats['forest_init_time'] = forest_timer.elapsed
        timing_stats['avg_seed_time'] = forest_timer.elapsed / args.num_seeds
        print(f"   ✅ 森林初始化完成，耗时: {forest_timer.get_elapsed_str()}")
        print(f"   📊 平均每个种子: {Timer.format_time(timing_stats['avg_seed_time'])}")
        
        # 执行搜索
        print(f"\n⏱️  [3/4] 搜索执行阶段 ({args.iterations} 次迭代)...")
        print(f"   预计时间: {Timer.format_time(timing_stats['avg_seed_time'] * args.iterations)}")
        
        with Timer("search") as search_timer:
            search.search(
                iterations=args.iterations,
                exploration_weight=args.exploration_weight,
                dataset_names=[args.dataset]
            )
        
        timing_stats['search_time'] = search_timer.elapsed
        timing_stats['avg_iteration_time'] = search_timer.elapsed / args.iterations
        print(f"   ✅ 搜索完成，耗时: {search_timer.get_elapsed_str()}")
        print(f"   📊 平均每次迭代: {Timer.format_time(timing_stats['avg_iteration_time'])}")
        
        # 保存结果
        print(f"\n⏱️  [4/4] 保存结果...")
        with Timer("save") as save_timer:
            timing_stats['save_time'] = 0  # ✅ 先设置占位值
            # 计算总时间
            timing_stats['total_time'] = time.time() - experiment_start_time
            timing_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            save_search_results(search, output_dir, vars(args), timing_stats)
        
        timing_stats['save_time'] = save_timer.elapsed
        print(f"   ✅ 结果保存完成，耗时: {save_timer.get_elapsed_str()}")
        
        # 打印最终结果（包含计时信息）
        print_final_results(search, timing_stats)
        
        # 打印总结
        total_time = time.time() - experiment_start_time
        end_datetime = datetime.now()
        
        print(f"\n✅ 实验完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"⏰ 结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总耗时: {Timer.format_time(total_time)}")
        
        # 估算如果运行更多迭代需要的时间
        if args.iterations < 100:
            estimated_100 = (timing_stats['avg_iteration_time'] * 100) + timing_stats['forest_init_time']
            print(f"\n💡 提示: 如果运行100次迭代，预计需要: {Timer.format_time(estimated_100)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验")
        timing_stats['total_time'] = time.time() - experiment_start_time
        timing_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timing_stats['interrupted'] = True
        print(f"⏱️  已运行时间: {Timer.format_time(timing_stats['total_time'])}")
        
    except Exception as e:
        print(f"\n❌ 搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        timing_stats['total_time'] = time.time() - experiment_start_time
        timing_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timing_stats['error'] = str(e)
        
    finally:
        # 确保工作进程被清理
        print("\n🧹 清理资源...")
        
        # 打印最终计时
        if 'total_time' not in timing_stats:
            timing_stats['total_time'] = time.time() - experiment_start_time
            timing_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    main()