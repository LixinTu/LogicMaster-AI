"""
DKT 模型训练脚本

Usage:
  python scripts/train_dkt.py                    # 默认训练
  python scripts/train_dkt.py --epochs 10        # 指定轮数
  python scripts/train_dkt.py --compare          # 包含 BKT 对比
  python scripts/train_dkt.py --force-numpy      # 强制使用 numpy 模型
  python scripts/train_dkt.py --force-lstm       # 强制使用 LSTM 模型

算法流程：
1. 从 answer_history 加载交互序列（按 user_id 分组）
2. 80/20 用户级划分训练/验证集
3. 自动选择模型（LSTM 或 numpy）
4. 训练 + 验证 + 早停
5. 保存权重到 models/
6. 输出训练报告到 reports/dkt_training_report.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_handler import DatabaseManager
from engine.skill_encoder import get_skill_encoder
from engine.dkt_model import (
    DKTModelNumpy,
    TORCH_AVAILABLE,
    MIN_INTERACTIONS_FOR_LSTM,
)

if TORCH_AVAILABLE:
    from engine.dkt_model import DKTModelLSTM


def load_sequences(db_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """从 answer_history 加载用户序列"""
    dm = DatabaseManager(db_path=db_path)
    rows = dm.query_answer_history()

    sequences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        user_id = row.get("user_id", "default")
        sequences[user_id].append({
            "question_id": row.get("question_id", ""),
            "skills": row.get("skill_ids", []),
            "is_correct": bool(row.get("is_correct", 0)),
            "theta_at_time": row.get("theta_at_time"),
        })

    return dict(sequences)


def compute_accuracy(
    model: Any,
    sequences: List[List[Dict[str, Any]]],
    encoder: Any = None,
) -> float:
    """计算预测准确率（predict > 0.5 vs actual）"""
    correct_count = 0
    total_count = 0

    for seq in sequences:
        for t in range(1, len(seq)):
            history = seq[:t]
            current = seq[t]
            skills = current.get("skills", [])
            is_correct = current.get("is_correct", False)

            if not isinstance(skills, list) or not skills:
                continue

            # 获取预测
            if hasattr(model, "predict_mastery"):
                if encoder is not None and hasattr(model, "num_skills"):
                    mastery = model.predict_mastery(history, encoder)
                else:
                    mastery = model.predict_mastery(history)
            else:
                continue

            # 对涉及的技能取平均掌握度
            preds = [mastery.get(s, 0.5) for s in skills]
            avg_pred = np.mean(preds) if preds else 0.5

            predicted_correct = avg_pred > 0.5
            if predicted_correct == is_correct:
                correct_count += 1
            total_count += 1

    return correct_count / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="DKT 模型训练")
    parser.add_argument("--db", default=os.path.join(PROJECT_ROOT, "logicmaster.db"), help="数据库路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心")
    parser.add_argument("--compare", action="store_true", help="输出 BKT 对比")
    parser.add_argument("--force-numpy", action="store_true", help="强制使用 numpy 模型")
    parser.add_argument("--force-lstm", action="store_true", help="强制使用 LSTM 模型")
    args = parser.parse_args()

    print("=" * 60)
    print("DKT Model Training")
    print("=" * 60)

    # 1. 构建/加载词表
    encoder = get_skill_encoder(args.db)
    print(f"Skills vocabulary: {encoder.num_skills} skills")
    if encoder.num_skills == 0:
        print("ERROR: No skills found in database. Exiting.")
        sys.exit(1)

    # 2. 加载序列
    user_sequences = load_sequences(args.db)
    total_interactions = sum(len(seq) for seq in user_sequences.values())
    print(f"Users: {len(user_sequences)}, Total interactions: {total_interactions}")

    if total_interactions < 10:
        print("WARNING: Less than 10 interactions. Not enough data for training.")
        print("Use the application to build up answer history first.")
        sys.exit(0)

    # 3. 用户级 80/20 分割
    user_ids = sorted(user_sequences.keys())
    n_train = max(1, int(len(user_ids) * 0.8))
    train_users = user_ids[:n_train]
    val_users = user_ids[n_train:] if n_train < len(user_ids) else []

    train_seqs = [user_sequences[u] for u in train_users]
    val_seqs = [user_sequences[u] for u in val_users]
    print(f"Train users: {len(train_users)}, Val users: {len(val_users)}")

    # 4. 模型选择
    use_lstm = (
        TORCH_AVAILABLE
        and total_interactions >= MIN_INTERACTIONS_FOR_LSTM
        and not args.force_numpy
    ) or args.force_lstm

    if args.force_lstm and not TORCH_AVAILABLE:
        print("ERROR: --force-lstm specified but torch is not available.")
        sys.exit(1)

    model_type = "lstm" if use_lstm else "numpy"
    print(f"Model type: {model_type}")

    # 5. 训练
    start_time = time.time()
    report: Dict[str, Any] = {
        "model_type": model_type,
        "num_skills": encoder.num_skills,
        "total_interactions": total_interactions,
        "num_users_train": len(train_users),
        "num_users_val": len(val_users),
    }

    if use_lstm:
        model = DKTModelLSTM(num_skills=encoder.num_skills)
        metrics = model.train_model(
            sequences=train_seqs,
            encoder=encoder,
            epochs=args.epochs,
            lr=0.001,
            patience=args.patience,
        )
        report["epochs_trained"] = len(metrics["train_losses"])
        report["best_epoch"] = metrics["best_epoch"]
        report["train_loss"] = metrics["train_losses"][-1] if metrics["train_losses"] else 0
        report["val_loss"] = metrics["val_losses"][-1] if metrics["val_losses"] else 0
        report["train_losses"] = metrics["train_losses"]
        report["val_losses"] = metrics["val_losses"]

        # 保存权重
        weights_path = os.path.join(PROJECT_ROOT, "models", "dkt_weights.pt")
        model.save_weights(weights_path)
        print(f"Weights saved: {weights_path}")

        # 计算准确率
        accuracy = compute_accuracy(model, val_seqs if val_seqs else train_seqs, encoder)
    else:
        model = DKTModelNumpy(db_path=args.db)
        metrics = model.train(sequences=train_seqs, epochs=args.epochs)
        report["epochs_trained"] = args.epochs
        report["best_epoch"] = args.epochs - 1
        report["train_loss"] = metrics["avg_loss"]
        report["val_loss"] = None
        report["num_updates"] = metrics["num_updates"]

        # 保存权重
        weights_path = os.path.join(PROJECT_ROOT, "models", "dkt_weights.pkl")
        model.save_weights(weights_path)
        print(f"Weights saved: {weights_path}")

        # 计算准确率
        accuracy = compute_accuracy(model, val_seqs if val_seqs else train_seqs)

    elapsed = time.time() - start_time
    report["overall_accuracy"] = accuracy
    report["training_time_seconds"] = round(elapsed, 2)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 6. 保存报告
    report_path = os.path.join(PROJECT_ROOT, "reports", "dkt_training_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")

    # 7. 输出摘要
    print("\n" + "-" * 40)
    print(f"Model:        {model_type}")
    print(f"Epochs:       {report['epochs_trained']}")
    print(f"Train loss:   {report['train_loss']:.4f}" if report['train_loss'] is not None else "Train loss:   N/A")
    if report.get("val_loss") is not None:
        print(f"Val loss:     {report['val_loss']:.4f}")
    print(f"Accuracy:     {accuracy:.2%}")
    print(f"Time:         {elapsed:.1f}s")
    print("-" * 40)

    # 8. BKT 对比
    if args.compare:
        print("\nBKT vs DKT Comparison:")
        print("-" * 60)
        # 使用所有训练序列合并为一个历史
        all_history = []
        for seq in train_seqs:
            all_history.extend(seq)

        if hasattr(model, "compare_with_bkt"):
            if use_lstm:
                comparison = model.compare_with_bkt(all_history, encoder)
            else:
                comparison = model.compare_with_bkt(all_history)

            print(f"{'Skill':<30} {'BKT Error':<12} {'DKT Mastery':<12} {'Agree?':<8}")
            print("-" * 62)
            for skill, data in sorted(comparison.items()):
                print(
                    f"{skill:<30} {data['bkt_error_rate']:<12.3f} "
                    f"{data['dkt_mastery']:<12.3f} {'Yes' if data['agreement'] else 'No':<8}"
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
