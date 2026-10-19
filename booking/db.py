"""
預約討論室資料庫層。

使用 SQLite 儲存預約與討論室資料。
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "campus.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # SQLite 外鍵約束預設不強制，需逐連線開啟
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """建立資料表並預設討論室資料。"""
    conn = _get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rooms (
                room_id   TEXT PRIMARY KEY,
                capacity  INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS reservations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id       TEXT NOT NULL,
                room_id          TEXT NOT NULL,
                people_count     INTEGER NOT NULL,
                reason           TEXT NOT NULL,
                start_time       TEXT NOT NULL,
                end_time         TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'confirmed',
                superseded_by_id INTEGER,
                created_at       TEXT NOT NULL,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id),
                FOREIGN KEY (superseded_by_id) REFERENCES reservations(id)
            );
            """
        )

        _migrate_add_superseded_by_id(conn)

        default_rooms = [
            ("A01", 4, 1),
            ("A02", 4, 1),
            ("A03", 4, 1),
            ("A04", 4, 1),
            ("A05", 4, 1),
            ("A06", 4, 1),
            ("B01", 6, 1),
            ("B02", 6, 1),
            ("B03", 6, 1),
            ("C01", 10, 1),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO rooms (room_id, capacity, is_active) VALUES (?, ?, ?)",
            default_rooms,
        )
        conn.commit()
    finally:
        conn.close()


def _migrate_add_superseded_by_id(conn: sqlite3.Connection) -> None:
    """對已存在的 reservations 表補上 superseded_by_id 欄位。"""
    cursor = conn.execute("PRAGMA table_info(reservations)")
    columns = [row[1] for row in cursor.fetchall()]
    if "superseded_by_id" not in columns:
        conn.execute("ALTER TABLE reservations ADD COLUMN superseded_by_id INTEGER")


def get_room(room_id: str) -> Optional[Dict]:
    """取得單一討論室，不存在則回傳 None。"""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM rooms WHERE room_id = ?", (room_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_available_rooms() -> List[Dict]:
    """列出所有開放中的討論室。"""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM rooms WHERE is_active = 1 ORDER BY room_id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def has_time_conflict(
    room_id: str,
    start_time: str,
    end_time: str,
    exclude_id: Optional[int] = None,
) -> bool:
    """
    檢查同一討論室在該時段是否已有重疊的有效預約。
    修改預約時可傳入 exclude_id 排除自己。
    """
    conn = _get_connection()
    try:
        query = """
            SELECT COUNT(*) AS cnt
            FROM reservations
            WHERE room_id = ?
              AND status = 'confirmed'
              AND start_time < ?
              AND end_time   > ?
        """
        params: List = [room_id, end_time, start_time]

        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)

        row = conn.execute(query, params).fetchone()
        return row["cnt"] > 0
    finally:
        conn.close()


def insert_reservation(
    student_id: str,
    room_id: str,
    people_count: int,
    reason: str,
    start_time: str,
    end_time: str,
    status: str = "confirmed",
) -> int:
    """寫入預約紀錄，回傳預約編號。"""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO reservations
                (student_id, room_id, people_count, reason,
                 start_time, end_time, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                room_id,
                people_count,
                reason,
                start_time,
                end_time,
                status,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_user_reservations(student_id: str, status: Optional[str] = None) -> List[Dict]:
    """查詢某使用者的預約，可依狀態過濾；未指定狀態則回傳全部。"""
    conn = _get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM reservations WHERE student_id = ? AND status = ? ORDER BY id",
                (student_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reservations WHERE student_id = ? ORDER BY id",
                (student_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_reservation(reservation_id: int) -> Optional[Dict]:
    """查詢單筆預約，不存在回傳 None。"""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cancel_reservation(reservation_id: int) -> bool:
    """取消預約：將狀態設為 'cancelled'。只對 'confirmed' 的預約生效。"""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "UPDATE reservations SET status = 'cancelled' "
            "WHERE id = ? AND status = 'confirmed'",
            (reservation_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def modify_reservation(old_id: int, new_data: Dict) -> Optional[int]:
    """
    修改預約：把舊預約設為 'superseded' 並指向新版本，同時新增一筆新預約。
    student_id 一律繼承舊預約，避免被改成他人。
    """
    conn = _get_connection()
    try:
        old = conn.execute(
            "SELECT * FROM reservations WHERE id = ? AND status = 'confirmed'",
            (old_id,),
        ).fetchone()
        if not old:
            return None

        cursor = conn.execute(
            """
            INSERT INTO reservations
                (student_id, room_id, people_count, reason,
                 start_time, end_time, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'confirmed', ?)
            """,
            (
                old["student_id"],
                new_data["room_id"],
                new_data["people_count"],
                new_data["reason"],
                new_data["start_time"],
                new_data["end_time"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        new_id = cursor.lastrowid

        conn.execute(
            "UPDATE reservations SET status = 'superseded', superseded_by_id = ? "
            "WHERE id = ?",
            (new_id, old_id),
        )

        conn.commit()
        return new_id
    except Exception:
        conn.rollback()
        return None
    finally:
        conn.close()
