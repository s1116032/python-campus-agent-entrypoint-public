"""
預約討論室業務邏輯層。

包含使用者身分取得、欄位驗證與預約送出。
"""

import re
from datetime import datetime
from typing import Dict, Optional

from booking.db import (
    cancel_reservation,
    get_reservation,
    get_room,
    has_time_conflict,
    insert_reservation,
    modify_reservation,
)

# 狀態中文映射，供錯誤訊息使用
STATUS_LABELS = {
    "confirmed": "已預約",
    "cancelled": "已取消",
    "superseded": "已被修改取代",
}

# 預設使用者 ID，後續將改為呼叫登入 API
DEFAULT_STUDENT_ID = "S123456789"

REASON_BLACKLIST = {"測試", "test", "asdf", "佔用", "隨便", "無", "null"}
REASON_MIN_LENGTH = 2
REASON_MAX_LENGTH = 100
TIME_FORMAT = "%Y-%m-%d %H:%M"

# 討論室開放時段（依知識庫）：週一至週五 08:00–21:30，週末 09:00–17:30
OPEN_HOURS = {
    "weekday": {"start": (8, 0), "end": (21, 30)},
    "weekend": {"start": (9, 0), "end": (17, 30)},
}


def get_current_user_id() -> str:
    """取得目前登入使用者的 student_id。"""
    return DEFAULT_STUDENT_ID


def _parse_time(time_str: str) -> Optional[datetime]:
    """解析時間字串，格式不符回傳 None。"""
    try:
        return datetime.strptime(time_str, TIME_FORMAT)
    except (ValueError, TypeError):
        return None


def _check_open_hours(start_dt: datetime, end_dt: datetime) -> Optional[str]:
    """檢查預約時段是否在開放時間內，回傳錯誤訊息；通過回傳 None。"""
    if start_dt.date() != end_dt.date():
        return "討論室預約不可跨天，開始與結束需在同一天。"

    is_weekend = start_dt.weekday() >= 5
    hours = OPEN_HOURS["weekend"] if is_weekend else OPEN_HOURS["weekday"]

    open_start = start_dt.replace(
        hour=hours["start"][0], minute=hours["start"][1], second=0, microsecond=0
    )
    open_end = start_dt.replace(
        hour=hours["end"][0], minute=hours["end"][1], second=0, microsecond=0
    )

    open_str = f"{hours['start'][0]:02d}:{hours['start'][1]:02d}–{hours['end'][0]:02d}:{hours['end'][1]:02d}"
    day_type = "週末" if is_weekend else "週一至週五"

    if start_dt < open_start:
        return f"開始時間需於開放時段內，{day_type}開放時間為 {open_str}。"

    if end_dt > open_end:
        return f"結束時間不得晚於閉館清點時間，{day_type}開放時間為 {open_str}。"

    return None


def validate_reason(reason: str) -> Optional[str]:
    """驗證事由，回傳錯誤訊息；通過回傳 None。"""
    if not isinstance(reason, str):
        return "事由格式不正確。"

    cleaned = re.sub(r"\s+", " ", reason).strip()

    if not cleaned:
        return "事由不可為空白。"
    if len(cleaned) < REASON_MIN_LENGTH:
        return f"事由至少需要 {REASON_MIN_LENGTH} 個字。"
    if len(cleaned) > REASON_MAX_LENGTH:
        return f"事由不可超過 {REASON_MAX_LENGTH} 個字。"
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned):
        return "事由需包含文字，不可只有數字或符號。"
    if cleaned.lower() in REASON_BLACKLIST:
        return "事由不可為測試用途，請填寫實際預約原因。"
    if re.fullmatch(r"([A-Za-z])\1*", cleaned):
        return "事由格式不正確，請填寫實際預約原因。"

    return None


def validate_booking(fields: Dict, exclude_id: Optional[int] = None) -> Optional[str]:
    """驗證預約欄位，回傳錯誤訊息；通過回傳 None。"""
    reason_error = validate_reason(fields.get("reason", ""))
    if reason_error:
        return reason_error

    room_id = fields.get("room_id", "")
    room = get_room(room_id)
    if not room:
        return f"找不到討論室 {room_id}。"
    if not room["is_active"]:
        return f"討論室 {room_id} 目前未開放預約。"

    people_count = fields.get("people_count")
    if not isinstance(people_count, int) or people_count < 2:
        return "討論室僅供多人使用，人數需為 2 人以上。單人自習請前往自習區。"
    if people_count > room["capacity"]:
        return f"人數超過討論室 {room_id} 的容量上限 {room['capacity']} 人。"

    start_time = fields.get("start_time", "")
    end_time = fields.get("end_time", "")
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time)
    if not start_dt or not end_dt:
        return f"時間格式不正確，請使用 {TIME_FORMAT} 格式。"
    if start_dt >= end_dt:
        return "結束時間需晚於開始時間。"
    if start_dt < datetime.now():
        return "不可預約過去的時間。"

    open_hours_error = _check_open_hours(start_dt, end_dt)
    if open_hours_error:
        return open_hours_error

    if has_time_conflict(room_id, start_time, end_time, exclude_id=exclude_id):
        return f"討論室 {room_id} 在該時段已有其他預約，請選擇其他時間。"

    return None


def submit_booking(fields: Dict) -> Dict:
    """送出預約。"""
    error = validate_booking(fields)
    if error:
        return {"success": False, "message": error, "reservation_id": None}

    student_id = get_current_user_id()
    reservation_id = insert_reservation(
        student_id=student_id,
        room_id=fields["room_id"],
        people_count=fields["people_count"],
        reason=fields["reason"],
        start_time=fields["start_time"],
        end_time=fields["end_time"],
    )

    return {
        "success": True,
        "message": f"預約成功！預約編號：{reservation_id}",
        "reservation_id": reservation_id,
    }


def _check_reservation_operable(
    reservation: Optional[Dict],
    reservation_id: int,
    action: str,
) -> Optional[str]:
    """檢查預約是否可操作。回傳錯誤訊息；可操作回傳 None。"""
    if not reservation:
        return f"找不到預約 #{reservation_id}。"

    if reservation["student_id"] != get_current_user_id():
        return f"預約 #{reservation_id} 不是您的預約。"

    if reservation["status"] == "superseded":
        new_id = reservation.get("superseded_by_id")
        hint = f"，請改用 #{new_id}" if new_id else ""
        return f"預約 #{reservation_id} 已被修改取代{hint}。"

    if reservation["status"] != "confirmed":
        label = STATUS_LABELS.get(reservation["status"], reservation["status"])
        return f"預約 #{reservation_id} 目前為「{label}」，無法{action}。"

    return None


def cancel_booking(reservation_id: int) -> Dict:
    """取消預約：將狀態設為 'cancelled'。"""
    reservation = get_reservation(reservation_id)

    error = _check_reservation_operable(reservation, reservation_id, "取消")
    if error:
        return {"success": False, "message": error, "reservation_id": None}

    if cancel_reservation(reservation_id):
        return {
            "success": True,
            "message": f"已取消預約 #{reservation_id}。",
            "reservation_id": reservation_id,
        }
    return {
        "success": False,
        "message": "取消失敗，請再試一次。",
        "reservation_id": None,
    }


def modify_booking(reservation_id: int, new_fields: Dict) -> Dict:
    """修改預約：軟刪除舊預約（設 'superseded'）並新增新預約。"""
    reservation = get_reservation(reservation_id)

    error = _check_reservation_operable(reservation, reservation_id, "修改")
    if error:
        return {"success": False, "message": error, "reservation_id": None}

    unchanged = (
        new_fields.get("room_id") == reservation["room_id"]
        and new_fields.get("people_count") == reservation["people_count"]
        and new_fields.get("reason") == reservation["reason"]
        and new_fields.get("start_time") == reservation["start_time"]
        and new_fields.get("end_time") == reservation["end_time"]
    )
    if unchanged:
        return {
            "success": False,
            "message": "沒有需要修改的內容。",
            "reservation_id": None,
        }

    validation_error = validate_booking(new_fields, exclude_id=reservation_id)
    if validation_error:
        return {"success": False, "message": validation_error, "reservation_id": None}

    new_id = modify_reservation(reservation_id, new_fields)
    if new_id:
        return {
            "success": True,
            "message": f"預約已修改，新預約編號為 #{new_id}。",
            "reservation_id": new_id,
        }
    return {
        "success": False,
        "message": "修改失敗，請再試一次。",
        "reservation_id": None,
    }
