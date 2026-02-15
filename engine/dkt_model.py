"""
Deep Knowledge Tracing (DKT) 模型

提供两种实现，运行时自动选择：
- DKTModelLSTM (Piech et al., 2015): PyTorch LSTM，需要 >= 50 条交互数据
- DKTModelNumpy: 滑动窗口逻辑回归，冷启动回退方案

自动选择逻辑（get_dkt_model）：
  1. torch 可用？（import 检查）
  2. answer_history >= 50 条交互？
  如果两者都满足 → DKTModelLSTM，否则 → DKTModelNumpy。
  随着数据积累，系统自动从 numpy 升级到 LSTM。

参考文献：
  Piech, C., et al. (2015). "Deep Knowledge Tracing."
  Advances in Neural Information Processing Systems (NeurIPS).
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np

from engine.skill_encoder import SkillEncoder, get_skill_encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
WINDOW_SIZE = 20           # 特征提取窗口大小
NUM_FEATURES = 4           # 每个技能的特征数（recency, cumulative_rate, streak, attempt_norm）
LEARNING_RATE = 0.05       # Numpy SGD 学习率
MIN_INTERACTIONS_FOR_LSTM = 50  # LSTM 启用阈值


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数"""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _extract_features(
    interactions: List[Dict[str, Any]], skill_name: str
) -> np.ndarray:
    """
    从交互历史中提取指定技能的特征矩阵。

    Args:
        interactions: 交互列表，每个包含 skills, is_correct
        skill_name: 目标技能名

    Returns:
        (WINDOW_SIZE, NUM_FEATURES) 特征矩阵
        4个特征: recency_weight, cumulative_correct_rate, streak, attempt_count_normalized
    """
    # 筛选包含该技能的交互
    relevant = []
    for inter in interactions:
        skills = inter.get("skills", [])
        if isinstance(skills, list) and skill_name in skills:
            relevant.append(inter)

    features = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)

    if not relevant:
        return features

    # 取最近 WINDOW_SIZE 条记录
    window = relevant[-WINDOW_SIZE:]
    total_relevant = len(relevant)
    running_correct = 0
    streak = 0

    for i, inter in enumerate(window):
        is_correct = inter.get("is_correct", False)

        # Feature 0: recency_weight（越近权重越大）
        features[i, 0] = (i + 1) / len(window)

        # Feature 1: cumulative_correct_rate（到当前为止的累计正确率）
        if is_correct:
            running_correct += 1
        features[i, 1] = running_correct / (i + 1)

        # Feature 2: streak（连续正确/错误次数，正确为正，错误为负，归一化到 [-1, 1]）
        if is_correct:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
        features[i, 2] = np.clip(streak / WINDOW_SIZE, -1.0, 1.0)

        # Feature 3: attempt_count_normalized（归一化尝试次数）
        features[i, 3] = min(total_relevant / 50.0, 1.0)

    return features


def _get_default_db_path() -> str:
    """项目根目录下的 logicmaster.db"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "logicmaster.db")


# ---------------------------------------------------------------------------
# DKTModelNumpy — 冷启动回退方案
# ---------------------------------------------------------------------------

class DKTModelNumpy:
    """
    滑动窗口逻辑回归 DKT（纯 numpy 实现）。

    每个技能维护独立的权重向量 (NUM_FEATURES,) + 偏置。
    用于冷启动或 PyTorch 不可用时。
    """

    def __init__(self, db_path: Optional[str] = None):
        self.encoder = get_skill_encoder(db_path)
        # 每个技能独立权重：{skill_name: (weights, bias)}
        self._weights: Dict[str, tuple] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """为每个已知技能初始化权重"""
        for skill in self.encoder.skill_to_id:
            if skill not in self._weights:
                w = np.zeros(NUM_FEATURES, dtype=np.float32)
                b = np.float32(0.0)
                self._weights[skill] = (w, b)

    def predict_mastery(
        self, student_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        预测各技能掌握概率。

        Args:
            student_history: 交互列表 [{skills, is_correct, ...}, ...]

        Returns:
            {skill_name: mastery_probability}
        """
        result: Dict[str, float] = {}

        for skill in self.encoder.skill_to_id:
            if skill not in self._weights:
                result[skill] = 0.5
                continue

            w, b = self._weights[skill]
            features = _extract_features(student_history, skill)

            if np.sum(np.abs(features)) < 1e-8:
                # 该技能无历史记录
                result[skill] = 0.5
                continue

            # 使用窗口均值特征
            mean_features = features.mean(axis=0)
            logit = float(np.dot(w, mean_features) + b)
            result[skill] = float(_sigmoid(np.array([logit]))[0])

        return result

    def train(
        self, sequences: List[List[Dict[str, Any]]], epochs: int = 1
    ) -> Dict[str, Any]:
        """
        在线 SGD 训练。

        Args:
            sequences: 用户序列列表，每个序列是交互列表
            epochs: 训练轮数

        Returns:
            {total_loss, num_updates, avg_loss}
        """
        total_loss = 0.0
        num_updates = 0

        for _ in range(epochs):
            for sequence in sequences:
                for t in range(1, len(sequence)):
                    current = sequence[t]
                    is_correct = current.get("is_correct", False)
                    skills = current.get("skills", [])
                    if not isinstance(skills, list):
                        continue

                    # 用 t 之前的历史作为输入
                    history = sequence[:t]

                    for skill in skills:
                        if skill not in self._weights:
                            continue

                        w, b = self._weights[skill]
                        features = _extract_features(history, skill)
                        mean_features = features.mean(axis=0)

                        # 前向传播
                        logit = float(np.dot(w, mean_features) + b)
                        pred = float(_sigmoid(np.array([logit]))[0])

                        # BCE 损失
                        target = 1.0 if is_correct else 0.0
                        eps = 1e-7
                        loss = -(
                            target * np.log(pred + eps)
                            + (1 - target) * np.log(1 - pred + eps)
                        )
                        total_loss += loss
                        num_updates += 1

                        # 梯度下降
                        grad = pred - target  # d(BCE)/d(logit)
                        w_new = w - LEARNING_RATE * grad * mean_features
                        b_new = b - LEARNING_RATE * grad
                        self._weights[skill] = (w_new, np.float32(b_new))

        avg_loss = total_loss / max(num_updates, 1)
        return {
            "total_loss": float(total_loss),
            "num_updates": num_updates,
            "avg_loss": float(avg_loss),
        }

    def compare_with_bkt(
        self, questions_log: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比 BKT（错误率）和 DKT 的技能评估。

        Returns:
            {skill: {bkt_error_rate, dkt_mastery, agreement}}
        """
        if not questions_log:
            return {}

        from engine.recommender import analyze_weak_skills

        # BKT 错误率计算
        skill_stats: Dict[str, Dict[str, int]] = {}
        for log in questions_log:
            skills = log.get("skills", [])
            is_correct = log.get("is_correct", False)
            if not isinstance(skills, list):
                continue
            for skill in skills:
                if skill not in skill_stats:
                    skill_stats[skill] = {"correct": 0, "total": 0}
                skill_stats[skill]["total"] += 1
                if is_correct:
                    skill_stats[skill]["correct"] += 1

        # DKT 预测
        dkt_mastery = self.predict_mastery(questions_log)

        result: Dict[str, Dict[str, Any]] = {}
        for skill, stats in skill_stats.items():
            total = stats["total"]
            error_rate = 1.0 - (stats["correct"] / total) if total > 0 else 0.5
            mastery = dkt_mastery.get(skill, 0.5)
            # agreement: 两者趋势一致（高错误率 → 低掌握度）
            agreement = abs((1.0 - error_rate) - mastery) < 0.3
            result[skill] = {
                "bkt_error_rate": error_rate,
                "dkt_mastery": mastery,
                "agreement": agreement,
            }

        return result

    def save_weights(self, filepath: str) -> None:
        """保存权重到 pickle 文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {}
        for skill, (w, b) in self._weights.items():
            data[skill] = {"weights": w.tolist(), "bias": float(b)}
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_weights(self, filepath: str) -> None:
        """从 pickle 文件加载权重"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        for skill, vals in data.items():
            w = np.array(vals["weights"], dtype=np.float32)
            b = np.float32(vals["bias"])
            self._weights[skill] = (w, b)


# ---------------------------------------------------------------------------
# DKTModelLSTM — PyTorch LSTM (Piech et al., 2015)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class DKTModelLSTM(nn.Module):
        """
        Deep Knowledge Tracing LSTM 模型 (Piech et al., 2015)。

        架构: LSTM(input_size=2K, hidden=64, layers=1) → Linear(K) → Sigmoid

        当 PyTorch 可用且训练数据 >= 50 条交互时自动启用。
        """

        def __init__(
            self,
            num_skills: int,
            hidden_size: int = 64,
            num_layers: int = 1,
        ):
            super().__init__()
            self.num_skills = num_skills
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.input_size = 2 * num_skills  # 正确通道 + 错误通道

            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output_layer = nn.Linear(hidden_size, num_skills)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            前向传播。

            Args:
                x: (batch, seq_len, 2K) 输入张量

            Returns:
                (batch, seq_len, K) 掌握概率
            """
            lstm_out, _ = self.lstm(x)
            logits = self.output_layer(lstm_out)
            return self.sigmoid(logits)

        def predict_mastery(
            self,
            student_history: List[Dict[str, Any]],
            encoder: Optional[SkillEncoder] = None,
        ) -> Dict[str, float]:
            """
            预测各技能掌握概率。

            Args:
                student_history: 交互列表
                encoder: 技能编码器

            Returns:
                {skill_name: mastery_probability}
            """
            if encoder is None:
                encoder = get_skill_encoder()

            if not student_history:
                return {s: 0.5 for s in encoder.skill_to_id}

            # 编码序列
            encoded = []
            for inter in student_history:
                skills = inter.get("skills", [])
                is_correct = inter.get("is_correct", False)
                if not isinstance(skills, list):
                    skills = []
                vec = encoder.encode_interaction(skills, is_correct)
                encoded.append(vec)

            # (1, seq_len, 2K)
            x = torch.tensor(np.array(encoded), dtype=torch.float32).unsqueeze(0)

            self.eval()
            with torch.no_grad():
                output = self.forward(x)  # (1, seq_len, K)
                last_output = output[0, -1, :].numpy()  # 取最后时间步

            return encoder.decode_predictions(last_output)

        def train_model(
            self,
            sequences: List[List[Dict[str, Any]]],
            encoder: Optional[SkillEncoder] = None,
            epochs: int = 10,
            lr: float = 0.001,
            patience: int = 5,
        ) -> Dict[str, Any]:
            """
            训练 LSTM 模型。

            Args:
                sequences: 用户序列列表
                encoder: 技能编码器
                epochs: 最大训练轮数
                lr: 学习率
                patience: 早停耐心

            Returns:
                {train_losses, val_losses, best_epoch, per_skill_auc}
            """
            if encoder is None:
                encoder = get_skill_encoder()

            # 80/20 用户级分割
            n_train = max(1, int(len(sequences) * 0.8))
            train_seqs = sequences[:n_train]
            val_seqs = sequences[n_train:] if n_train < len(sequences) else []

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            criterion = nn.BCELoss()

            train_losses: List[float] = []
            val_losses: List[float] = []
            best_val_loss = float("inf")
            best_epoch = 0
            best_state = None
            patience_counter = 0

            for epoch in range(epochs):
                # --- 训练 ---
                self.train()
                epoch_loss = 0.0
                n_batches = 0

                for seq in train_seqs:
                    if len(seq) < 2:
                        continue

                    # 编码输入序列（t=0..T-2）和目标（t=1..T-1）
                    inputs = []
                    targets = []
                    for t in range(len(seq) - 1):
                        skills = seq[t].get("skills", [])
                        is_correct = seq[t].get("is_correct", False)
                        if not isinstance(skills, list):
                            skills = []
                        inputs.append(encoder.encode_interaction(skills, is_correct))

                        # 目标：下一时间步各技能的正确概率
                        next_skills = seq[t + 1].get("skills", [])
                        next_correct = seq[t + 1].get("is_correct", False)
                        target_vec = np.zeros(encoder.num_skills, dtype=np.float32)
                        if isinstance(next_skills, list):
                            for s in next_skills:
                                idx = encoder.skill_to_id.get(s)
                                if idx is not None:
                                    target_vec[idx] = 1.0 if next_correct else 0.0
                        targets.append(target_vec)

                    if not inputs:
                        continue

                    x = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(0)
                    y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(0)

                    optimizer.zero_grad()
                    output = self.forward(x)  # (1, T-1, K)

                    # 只计算有技能标注的位置的损失
                    mask = (y.sum(dim=-1, keepdim=True) != 0).float().expand_as(y)
                    # 避免无标注位置影响：用 mask 加权
                    if mask.sum() > 0:
                        loss = criterion(output * mask, y * mask)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1

                avg_train_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_train_loss)

                # --- 验证 ---
                if val_seqs:
                    self.eval()
                    val_loss = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for seq in val_seqs:
                            if len(seq) < 2:
                                continue
                            inputs = []
                            targets = []
                            for t in range(len(seq) - 1):
                                skills = seq[t].get("skills", [])
                                is_correct = seq[t].get("is_correct", False)
                                if not isinstance(skills, list):
                                    skills = []
                                inputs.append(encoder.encode_interaction(skills, is_correct))
                                next_skills = seq[t + 1].get("skills", [])
                                next_correct = seq[t + 1].get("is_correct", False)
                                target_vec = np.zeros(encoder.num_skills, dtype=np.float32)
                                if isinstance(next_skills, list):
                                    for s in next_skills:
                                        idx = encoder.skill_to_id.get(s)
                                        if idx is not None:
                                            target_vec[idx] = 1.0 if next_correct else 0.0
                                targets.append(target_vec)

                            if not inputs:
                                continue
                            x = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(0)
                            y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(0)
                            output = self.forward(x)
                            mask = (y.sum(dim=-1, keepdim=True) != 0).float().expand_as(y)
                            if mask.sum() > 0:
                                loss = criterion(output * mask, y * mask)
                                val_loss += loss.item()
                                n_val += 1

                    avg_val_loss = val_loss / max(n_val, 1)
                    val_losses.append(avg_val_loss)

                    # 早停
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch
                        best_state = {k: v.clone() for k, v in self.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                else:
                    # 无验证集：保存最后一轮
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}

            # 恢复最佳权重
            if best_state is not None:
                self.load_state_dict(best_state)

            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "per_skill_auc": {},  # 由 training script 计算
            }

        def compare_with_bkt(
            self,
            questions_log: List[Dict[str, Any]],
            encoder: Optional[SkillEncoder] = None,
        ) -> Dict[str, Dict[str, Any]]:
            """对比 BKT 和 DKT 的技能评估"""
            if not questions_log:
                return {}

            if encoder is None:
                encoder = get_skill_encoder()

            # BKT 错误率
            skill_stats: Dict[str, Dict[str, int]] = {}
            for log in questions_log:
                skills = log.get("skills", [])
                is_correct = log.get("is_correct", False)
                if not isinstance(skills, list):
                    continue
                for skill in skills:
                    if skill not in skill_stats:
                        skill_stats[skill] = {"correct": 0, "total": 0}
                    skill_stats[skill]["total"] += 1
                    if is_correct:
                        skill_stats[skill]["correct"] += 1

            # DKT 预测
            dkt_mastery = self.predict_mastery(questions_log, encoder)

            result: Dict[str, Dict[str, Any]] = {}
            for skill, stats in skill_stats.items():
                total = stats["total"]
                error_rate = 1.0 - (stats["correct"] / total) if total > 0 else 0.5
                mastery = dkt_mastery.get(skill, 0.5)
                agreement = abs((1.0 - error_rate) - mastery) < 0.3
                result[skill] = {
                    "bkt_error_rate": error_rate,
                    "dkt_mastery": mastery,
                    "agreement": agreement,
                }

            return result

        def save_weights(self, filepath: str) -> None:
            """保存模型权重"""
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            torch.save(self.state_dict(), filepath)

        def load_weights(self, filepath: str, num_skills: Optional[int] = None) -> None:
            """加载模型权重"""
            state = torch.load(filepath, weights_only=True)
            self.load_state_dict(state)


# ---------------------------------------------------------------------------
# 自动选择
# ---------------------------------------------------------------------------

def get_dkt_model(
    db_path: Optional[str] = None,
) -> Union[DKTModelNumpy, Any]:
    """
    自动选择 DKT 模型。

    选择逻辑：
    1. torch 可用？（import 检查）
    2. answer_history >= 50 条交互？
    如果两者都满足 → DKTModelLSTM，否则 → DKTModelNumpy。

    Args:
        db_path: 数据库路径

    Returns:
        DKTModelNumpy 或 DKTModelLSTM 实例
    """
    db_path = db_path or _get_default_db_path()

    if TORCH_AVAILABLE:
        # 检查交互数量
        try:
            from utils.db_handler import DatabaseManager
            dm = DatabaseManager(db_path=db_path)
            count = dm.count_answer_history()
            if count >= MIN_INTERACTIONS_FOR_LSTM:
                encoder = get_skill_encoder(db_path)
                if encoder.num_skills > 0:
                    model = DKTModelLSTM(num_skills=encoder.num_skills)
                    logger.info(
                        f"DKT: selected LSTM model ({count} interactions, {encoder.num_skills} skills)"
                    )
                    return model
        except Exception as e:
            logger.warning(f"DKT LSTM check failed, falling back to numpy: {e}")

    model = DKTModelNumpy(db_path=db_path)
    logger.info("DKT: selected numpy model (cold-start or torch unavailable)")
    return model
