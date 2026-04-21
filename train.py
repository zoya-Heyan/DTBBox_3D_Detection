import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.kitti_dataset import KittiDataset
from models.dtbbox_net import DTBoxNet
from utils.misc import load_config, save_checkpoint
from utils.losses import compute_loss

def main():
    parser = argparse.ArgumentParser(description="Train DT-BBox 3D Detection")
    parser.add_argument("--config", type=str, default="configs/kitti.yaml", help="Config file path")
    parser.add_argument("--stage", type=str, default="baseline", choices=["baseline", "dtbbox", "relative", "full"], help="Training stage")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    config.stage = args.stage

    # 数据集
    train_dataset = KittiDataset(
        root=config.dataset.root,
        split_file=config.dataset.train_split,
        input_size=config.dataset.input_size,
        allowed_classes=config.dataset.allowed_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=lambda batch: batch
    )

    # 模型
    model = DTBoxNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.train.lr,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    # 训练
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # 准备数据
            images = []
            bboxes = []
            targets = []

            for sample in batch:
                images.append(sample["image"].to(device))
                bboxes.append(sample["bboxes"].to(device))
                targets.append({
                    "locations": sample["locations"].to(device),
                    "dimensions": sample["dimensions"].to(device),
                    "yaws": sample["yaws"].to(device)
                })

            images = torch.stack(images)

            # 前向传播
            preds = model(images, bboxes)

            # 计算损失
            loss = compute_loss(preds, targets, config)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

        # 打印日志
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.train.epochs}, Loss: {avg_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, config.train.save_dir)

if __name__ == "__main__":
    main()