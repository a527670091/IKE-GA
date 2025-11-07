import multiprocessing
import os
import time
from datetime import datetime
from dotenv import load_dotenv

from evo_ice.evo_ice import evo_ice_main, parse_args, save_final_results
from evo_ice.plotting import plot_performance_curves

if __name__ == '__main__':
    # This is a fix for a multiprocessing issue on macOS and other systems
    # where the 'fork' start method is the default. 'spawn' is more stable.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # The start method can only be set once.
        pass

    # 1. 自动从 .env 文件加载环境变量
    load_dotenv()

    # 2. 强制将代理设置写入当前运行环境
    # 这是最主流和可靠的方式，确保所有子进程都能继承。
    proxy_url = os.getenv("HTTPS_PROXY")
    if proxy_url:
        print(f"--- 代理配置 ---")
        print(f"检测到代理: {proxy_url}")
        print("正在强制为当前脚本运行环境设置 HTTP_PROXY 和 HTTPS_PROXY...")
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        print("代理设置完毕。")
        print("--------------------")

    # --- 调试步骤 ---
    # 检查API密钥是否已成功加载
    google_api_key = os.getenv("GOOGLE_API_KEY")
    print("--- 调试信息 ---")
    if google_api_key:
        print(f"GOOGLE_API_KEY 加载成功!")
        # 为了安全，我们只打印密钥的一小部分来确认
        print(f"加载到的密钥 (部分): {google_api_key[:4]}...{google_api_key[-4:]}")
    else:
        print(f"GOOGLE_API_KEY 加载失败!")
        print("未能从 .env 文件或环境中加载 GOOGLE_API_KEY。")
        print("请您确认以下两点：")
        print("1. 在项目的根目录下，确实存在一个名为 '.env' 的文件 (而不是 '.env.example')。")
        print("2. 打开 '.env' 文件，确认里面有一行 `GOOGLE_API_KEY=...`，并且等号两边没有多余的空格或引号。")
    print("--------------------")
    # --- 调试结束 ---

    start_time = time.time()
    args = parse_args()
    
    # --- 为本次运行创建唯一的输出目录 ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_name = f"{timestamp}_seed{args.seed}_pop{args.population_size}_gens{args.num_generations}_k{args.k_demos}"
    run_output_dir = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"\n--- 实验输出 ---")
    print(f"本次运行的结果将保存在: {run_output_dir}")
    print("--------------------")

    # evo_ice_main 现在返回 population, fitness_scores 和 history
    final_population, final_fitness_scores, history = evo_ice_main(args)

    # 1. 将JSON结果文件保存到新目录中
    save_final_results(args, final_population, final_fitness_scores, start_time, run_output_dir)

    # 2. 将性能曲线图绘制并保存到新目录中
    if history:
        plot_performance_curves(history, run_output_dir)
