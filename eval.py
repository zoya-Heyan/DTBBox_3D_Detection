import argparse
import torch
from torch.utils.data import DataLoader
from datasets.kitti_dataset import KittiDataset
from models.dtbbox_net import DTBoxNet
from utils.misc import load_config, load_checkpoint
from utils.metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate DT-BBox 3D Detection")
    parser.add_argument("--config", type=str, default="configs/kitti.yaml", help="Config file path")
    parser.add_argument("--stage", type=str, default="baseline", choices=["baseline", "dtbbox", "relative", "full"], help="Evaluation stage")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    config.stage = args.stage

    # 数据集
    val_dataset = KittiDataset(
        root=config.dataset.root,
        split_file=config.dataset.val_split,
        input_size=config.dataset.input_size,
        allowed_classes=config.dataset.allowed_classes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=lambda batch: batch
    )

    # 模型
    model = DTBoxNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载检查点
    load_checkpoint(model, None, args.checkpoint)

    # 评估
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # 准备数据
            images = []
            bboxes = []
            calibs = []
            targets = []

            for sample in batch:
                images.append(sample["image"].to(device))
                bboxes.append(sample["bboxes"].to(device))
                calibs.append(sample["calib"].to(device))
                targets.append({
                    "locations": sample["locations"].to(device),
                    "dimensions": sample["dimensions"].to(device),
                    "yaws": sample["yaws"].to(device)
                })

            images = torch.stack(images)

            # 推理
            preds = model.inference(images, bboxes, calibs)

            all_preds.extend(preds)
            all_targets.extend(targets)

    # 计算指标
    metrics = evaluate(all_preds, all_targets)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()