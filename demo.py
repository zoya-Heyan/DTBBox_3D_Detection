import argparse
import cv2
import torch
import os
from datasets.kitti_dataset import KittiDataset
from models.dtbbox_net import DTBoxNet
from utils.misc import load_config, load_checkpoint
from utils.visualize import visualize_prediction

def main():
    parser = argparse.ArgumentParser(description="Demo DT-BBox 3D Detection")
    parser.add_argument("--config", type=str, default="configs/kitti.yaml", help="Config file path")
    parser.add_argument("--stage", type=str, default="baseline", choices=["baseline", "dtbbox", "relative", "full"], help="Demo stage")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--sample_id", type=str, required=True, help="Sample ID")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    config.stage = args.stage

    # 模型
    model = DTBoxNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载检查点
    load_checkpoint(model, None, args.checkpoint)

    # 加载样本
    dataset = KittiDataset(
        root=config.dataset.root,
        split_file=config.dataset.val_split,
        input_size=config.dataset.input_size,
        allowed_classes=config.dataset.allowed_classes
    )

    # 查找样本
    sample_idx = None
    for i, sample_id in enumerate(dataset.sample_ids):
        if sample_id == args.sample_id:
            sample_idx = i
            break

    if sample_idx is None:
        print(f"Sample {args.sample_id} not found")
        return

    sample = dataset[sample_idx]

    # 准备数据
    image = sample["image"].unsqueeze(0).to(device)
    bboxes = [sample["bboxes"].to(device)]
    calib = sample["calib"].to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        preds = model.inference(image, bboxes, [calib])[0]

    # 可视化
    img = cv2.imread(os.path.join(config.dataset.root, "training", "image_2", f"{args.sample_id}.png"))
    img = visualize_prediction(img, preds, sample["calib"].numpy())

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.sample_id}_{args.stage}.png")
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()