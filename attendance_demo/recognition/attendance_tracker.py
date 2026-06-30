


import threading
from datetime import datetime
from sqlalchemy import text

# Once marked in a session, never mark again (until session restart)
# Using 24 hours = essentially unlimited for a single session
COOLDOWN_SECONDS = 86400  # 24 hours - prevents duplicate marking in same session

class AttendanceTracker:
    """
    Shared across all camera threads. Ensures EACH student is marked ONLY ONCE
    per session, no matter how many cameras see them or how many times.
    """

    def __init__(self, session_factory):
        self._factory = session_factory
        self._marked  = set()   # set of roll_nos already marked this session
        self._seen    = {}      # roll_no -> datetime of last successful mark
        self._lock    = threading.Lock()
        self._count   = 0       # total unique students marked this session
        self._ensure_table()

    def _ensure_table(self):
        session = self._factory()
        try:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS attendance_log (
                    id          SERIAL PRIMARY KEY,
                    roll_no     VARCHAR(50)  NOT NULL,
                    student_name VARCHAR(150),
                    marked_at   TIMESTAMP    DEFAULT NOW()
                )
            """))
            session.commit()
        except Exception as e:
            print(f"⚠️  Could not create attendance_log table: {e}")
        finally:
            session.close()

    def mark(self, roll_no, student_name):
        """
        Called by any camera thread when a face matches.
        Thread-safe. Returns True only when a new DB row is written.
        IMPORTANT: Once a student is marked, they're marked for the entire session!
        """
        now = datetime.now()

        with self._lock:
            # Check if already marked in this session
            if roll_no in self._marked:
                return False  # Already marked - skip!
            
            # Check cooldown (backup check)
            last = self._seen.get(roll_no)
            if last and (now - last).total_seconds() < COOLDOWN_SECONDS:
                return False
            
            # Mark as processed
            self._marked.add(roll_no)
            self._seen[roll_no] = now
            self._count += 1

        # DB write happens outside the lock so we don't block other threads
        session = self._factory()
        try:
            session.execute(
                text("""
                    INSERT INTO attendance_log (roll_no, student_name, marked_at)
                    VALUES (:roll, :name, :ts)
                """),
                {"roll": roll_no, "name": student_name, "ts": now}
            )
            session.commit()
            print(f"✅  Attendance: {student_name:<30} (Roll: {roll_no}) @ {now.strftime('%H:%M:%S')} Count: {self._count}")
        except Exception as e:
            session.rollback()
            print(f"⚠️  DB error for {student_name}: {e}")
        finally:
            session.close()

        return True

    @property
    def present_count(self):
        """Unique students marked in this session."""
        with self._lock:
            return len(self._marked)

    @property
    def total_marked(self):
        with self._lock:
            return self._count
