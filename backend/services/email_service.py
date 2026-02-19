"""
邮件提醒服务：间隔重复复习提醒
基于 Half-Life Regression 遗忘曲线，向有待复习题目的用户发送 HTML 邮件
"""

import smtplib
import sqlite3
import os
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from engine.spaced_repetition import SpacedRepetitionModel


# ---------- HTML 模板 ----------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }}
    .container {{ max-width: 600px; margin: 0 auto; }}
    .header {{ border-bottom: 2px solid #21262d; padding-bottom: 16px; margin-bottom: 24px; }}
    .logo {{ font-size: 24px; font-weight: bold; color: #58a6ff; letter-spacing: 2px; }}
    .tagline {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
    .alert-box {{ background: #161b22; border: 1px solid #f85149; border-left: 4px solid #f85149;
                  border-radius: 6px; padding: 20px; margin: 20px 0; }}
    .alert-title {{ font-size: 20px; font-weight: bold; color: #f85149; margin-bottom: 8px; }}
    .alert-body {{ font-size: 14px; color: #c9d1d9; line-height: 1.6; }}
    .question-list {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px;
                      padding: 16px; margin: 16px 0; }}
    .question-item {{ border-bottom: 1px solid #21262d; padding: 10px 0; }}
    .question-item:last-child {{ border-bottom: none; }}
    .recall-bar-bg {{ background: #21262d; border-radius: 4px; height: 6px; margin-top: 6px; }}
    .recall-bar {{ background: #f85149; border-radius: 4px; height: 6px; }}
    .cta {{ text-align: center; margin: 24px 0; }}
    .cta-btn {{ background: #58a6ff; color: #0d1117; padding: 12px 28px; border-radius: 6px;
                text-decoration: none; font-weight: bold; font-size: 14px; display: inline-block; }}
    .footer {{ font-size: 11px; color: #8b949e; text-align: center; margin-top: 32px;
               border-top: 1px solid #21262d; padding-top: 16px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo">🧠 GlitchMind</div>
    <div class="tagline">AI-Native Adaptive GMAT Learning</div>
  </div>

  <div class="alert-box">
    <div class="alert-title">⚡ Memory Signal Degrading</div>
    <div class="alert-body">
      Hi {user_name},<br><br>
      You have <strong>{due_count} question{plural}</strong> losing signal.
      Time to debug your memory before the forgetting curve wins.
    </div>
  </div>

  <p style="color:#8b949e; font-size:13px;">Most urgent reviews (lowest recall):</p>
  <div class="question-list">
    {question_rows}
  </div>

  <div class="cta">
    <a href="http://localhost:8501" class="cta-btn">▶ Start Review Session</a>
  </div>

  <div class="footer">
    GlitchMind · AI-powered GMAT prep · You received this because review reminders are enabled.<br>
    Unsubscribe by removing your email from the platform settings.
  </div>
</div>
</body>
</html>
"""

_QUESTION_ROW_TEMPLATE = """\
<div class="question-item">
  <strong style="color:#58a6ff;">#{rank}.</strong>
  <span style="color:#8b949e; font-size:12px;">Question {q_id}</span>
  <div style="font-size:12px; color:#8b949e; margin-top:4px;">
    Recall: <span style="color:{recall_color};">{recall_pct}%</span>
    &nbsp;·&nbsp; Elapsed: {elapsed:.1f}d &nbsp;·&nbsp; Half-life: {half_life:.1f}d
  </div>
  <div class="recall-bar-bg">
    <div class="recall-bar" style="width:{recall_pct}%;"></div>
  </div>
</div>
"""


class EmailReminderService:
    """
    间隔重复邮件提醒服务。
    SMTP 配置缺失时，所有 send_* 方法静默降级（返回 False）。
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        db_path: Optional[str] = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.db_path = db_path or self._default_db_path()
        self._configured = bool(smtp_host and sender_email and sender_password)

    @staticmethod
    def _default_db_path() -> str:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return os.path.join(project_root, "logicmaster.db")

    # ---------- 公开方法 ----------

    def send_review_reminder(
        self,
        to_email: str,
        user_name: str,
        due_count: int,
        questions_preview: List[Dict[str, Any]],
    ) -> bool:
        """
        向用户发送复习提醒 HTML 邮件。

        Args:
            to_email: 收件人邮箱
            user_name: 收件人显示名称
            due_count: 待复习题目总数
            questions_preview: 最多 3 道最急迫复习题 [{question_id, recall_probability, half_life, elapsed_days}]

        Returns:
            发送成功返回 True，SMTP 未配置或失败返回 False
        """
        if not self._configured:
            return False

        # 构建 question rows HTML
        rows_html = ""
        for i, q in enumerate(questions_preview[:3], 1):
            recall = q.get("recall_probability", 0.0)
            recall_pct = round(recall * 100)
            recall_color = "#f85149" if recall < 0.3 else "#e3b341" if recall < 0.5 else "#3fb950"
            rows_html += _QUESTION_ROW_TEMPLATE.format(
                rank=i,
                q_id=q.get("question_id", "?")[:16],
                recall_color=recall_color,
                recall_pct=recall_pct,
                elapsed=q.get("elapsed_days", 0.0),
                half_life=q.get("half_life", 1.0),
            )

        if not rows_html:
            rows_html = "<div style='color:#8b949e; font-size:13px;'>（无具体题目信息）</div>"

        html_body = _HTML_TEMPLATE.format(
            user_name=user_name or to_email.split("@")[0],
            due_count=due_count,
            plural="s" if due_count != 1 else "",
            question_rows=rows_html,
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🧠 [{due_count} reviews due] GlitchMind Memory Alert"
        msg["From"] = self.sender_email
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, to_email, msg.as_string())
            return True
        except Exception as e:
            print(f"send_review_reminder failed ({to_email}): {e}")
            return False

    def check_and_send_reminders(self) -> int:
        """
        遍历所有有间隔重复数据的注册用户，向有到期复习且 24h 内未提醒的用户发送邮件。

        Returns:
            实际发送的邮件数
        """
        if not self._configured:
            print("SMTP 未配置，跳过邮件发送")
            return 0

        # 获取所有有间隔重复数据的用户
        users_with_reviews = self._get_users_with_due_reviews()
        sent_count = 0

        for user_info in users_with_reviews:
            user_id = user_info["user_id"]
            email = user_info.get("email")
            display_name = user_info.get("display_name", "")
            due_candidates = user_info["due_candidates"]

            if not email:
                continue

            # 检查 24h 内是否已发过提醒
            last_sent = self._get_last_reminder_time(user_id)
            if last_sent:
                try:
                    last_dt = datetime.fromisoformat(last_sent)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) - last_dt < timedelta(hours=24):
                        continue  # 24h 内已发送，跳过
                except (ValueError, TypeError):
                    pass

            # 发送提醒
            success = self.send_review_reminder(
                to_email=email,
                user_name=display_name or email.split("@")[0],
                due_count=len(due_candidates),
                questions_preview=due_candidates[:3],
            )
            if success:
                self._log_email_sent(user_id)
                sent_count += 1
                print(f"  ✓ Sent reminder to {email} ({len(due_candidates)} reviews due)")

        return sent_count

    # ---------- 私有方法 ----------

    def _get_users_with_due_reviews(self) -> List[Dict[str, Any]]:
        """
        查询所有在 spaced_repetition_stats 中有 recall < 0.5 题目的注册用户。
        通过 users 表获取邮箱。
        """
        conn = None
        results = []
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            # 获取所有有间隔重复数据的 user_id（只处理有注册邮箱的用户）
            cursor.execute(
                """SELECT DISTINCT srs.user_id, u.email, u.display_name
                   FROM spaced_repetition_stats srs
                   LEFT JOIN users u ON srs.user_id = u.id
                   WHERE u.email IS NOT NULL""",
            )
            user_rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            if conn:
                conn.close()
            print(f"_get_users_with_due_reviews query failed: {e}")
            return []

        now = datetime.now(timezone.utc)
        for user_id, email, display_name in user_rows:
            try:
                sr = SpacedRepetitionModel(db_path=self.db_path, user_id=user_id)
                candidates = sr.get_review_candidates(threshold=0.5)
                if candidates:
                    results.append({
                        "user_id": user_id,
                        "email": email,
                        "display_name": display_name or "",
                        "due_candidates": candidates,
                    })
            except Exception as e:
                print(f"  SR query failed for user {user_id}: {e}")

        return results

    def _get_last_reminder_time(self, user_id: str) -> Optional[str]:
        """从 email_logs 获取最后一次提醒时间"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT MAX(sent_at) FROM email_logs
                   WHERE user_id = ? AND email_type = 'review_reminder'""",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            if conn:
                conn.close()
            return None

    def _log_email_sent(self, user_id: str) -> None:
        """记录邮件发送日志"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                "INSERT INTO email_logs (user_id, email_type) VALUES (?, 'review_reminder')",
                (user_id,),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            if conn:
                conn.close()
            print(f"_log_email_sent failed: {e}")


# ---------- 工厂函数 ----------

def get_email_service(db_path: Optional[str] = None) -> EmailReminderService:
    """
    从 backend.config 读取 SMTP 配置，构建 EmailReminderService 实例。
    """
    try:
        from backend.config import settings
        return EmailReminderService(
            smtp_host=settings.SMTP_HOST,
            smtp_port=settings.SMTP_PORT,
            sender_email=settings.SMTP_EMAIL,
            sender_password=settings.SMTP_PASSWORD,
            db_path=db_path,
        )
    except ImportError:
        return EmailReminderService(
            smtp_host="", smtp_port=587,
            sender_email="", sender_password="",
            db_path=db_path,
        )
