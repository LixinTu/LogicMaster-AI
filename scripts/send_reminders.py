"""
间隔重复复习提醒发送脚本

用法：
    python scripts/send_reminders.py

可通过 cron / Task Scheduler 每日定时调用：
    # Linux cron (daily at 9am)
    0 9 * * * /path/to/venv/bin/python /path/to/scripts/send_reminders.py

    # Windows Task Scheduler: 指向此脚本，每天 09:00 触发
"""

import os
import sys
from datetime import datetime

# 将项目根目录加入 Python 路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.services.email_service import get_email_service


def main() -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GlitchMind Review Reminder Job")
    print("=" * 60)

    service = get_email_service()

    if not service._configured:
        print("⚠  SMTP 未配置。请在 .env 中设置 SMTP_HOST / SMTP_EMAIL / SMTP_PASSWORD。")
        print("   跳过邮件发送。")
        return

    print("🔍 Scanning users with due spaced-repetition reviews...")
    sent = service.check_and_send_reminders()

    print("=" * 60)
    print(f"✅ Done. Sent reminders to {sent} user(s).")


if __name__ == "__main__":
    main()
