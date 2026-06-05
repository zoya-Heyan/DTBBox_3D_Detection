import yaml
import os
import torch

class Config:
    """
    配置类
    """
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __getattr__(self, name):
        return None

def load_config(config_path):
    """
    加载配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        config: 配置对象
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # 递归转换为 Config 对象
    def convert_to_config(obj):
        if isinstance(obj, dict):
            config = Config({})
            for key, value in obj.items():
                setattr(config, key, convert_to_config(value))
            return config
        elif isinstance(obj, list):
            return [convert_to_config(item) for item in obj]
        else:
            return obj

    return convert_to_config(config_dict)

def save_checkpoint(model, optimizer, epoch, save_dir):
    """
    保存模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        epoch:  epoch
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
    Returns:
        epoch: 加载的 epoch
    """
    checkpoint = torch.load(checkpoint_path)
    
    # 使用 strict=False 允许加载不同阶段的检查点（忽略不存在的键）
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict, strict=False)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"]