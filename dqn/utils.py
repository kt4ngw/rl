import torch
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import random


def set_seed(seed):
    """设置固定的随机种子，确保训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    torch.backends.cudnn.deterministic = True  # 确保CUDA有确定性结果
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化


def save_model(model, episode, path):
    """保存模型检查点"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存 at episode {episode}")

def plot_rewards(rewards, save_path, window=10):
    """绘制奖励的滚动平均曲线"""
    moving_avg = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(len(rewards))]
    plt.plot(rewards, label="Rewards")
    plt.plot(moving_avg, label="Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    # 检查保存路径中的文件夹是否存在，如果不存在则创建
    folder = os.path.dirname(save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 保存图像到指定路径
    plt.savefig(save_path)
    print(f"图像已保存到: {save_path}")
    
    # 显示图像
    plt.show()

import datetime
def setup_logger():
    # 获取当前时间并格式化为字符串
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建日志文件保存的目录
    log_dir = "./log/dqn"
    os.makedirs(log_dir, exist_ok=True)  # 创建目录，如果目录已存在则不报错

    # 创建带有时间戳的日志文件名
    log_file = f"{log_dir}/train_log_{timestamp}.log"

    # 创建文件处理器，指定日志文件路径
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 获取 logger 对象并设置日志级别
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger