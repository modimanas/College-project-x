-- Registration / core hierarchy schema (PostgreSQL + pgvector)
-- Run once: psql -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Hierarchy ────────────────────────────────────────────────────────────
--   branches  →  sections  →  students  →  embeddings
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS branches (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code          TEXT UNIQUE NOT NULL,  -- 'CSIT', 'CSE', 'ECE', 'ME', ...
    name          TEXT NOT NULL,         -- full display name shown to students
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Each section belongs to exactly one branch + year (e.g. CSIT / Year 2 / Section A).
-- Detection cameras are mapped to sections; they only search this section's FAISS index.
CREATE TABLE IF NOT EXISTS sections (
    id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    branch_id UUID NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    name      TEXT NOT NULL,              -- 'A', 'B', 'C', 'D'
    year      SMALLINT NOT NULL CHECK (year BETWEEN 1 AND 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (branch_id, name, year)
);

CREATE TABLE IF NOT EXISTS rooms (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT UNIQUE NOT NULL,
    section_id  UUID REFERENCES sections(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,              -- 'cam_001'
    room_id     UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    position    TEXT,                          -- 'left' | 'center' | 'right'
    rtsp_url    TEXT
);

-- Enforce one camera per position per room.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_room_camera_position
    ON cameras (room_id, position);

-- Enforce the allowed position labels.
DO $$
BEGIN
    ALTER TABLE cameras
        ADD CONSTRAINT cameras_position_check
        CHECK (position IN ('left', 'center', 'right'));
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

-- Enforce exactly 3 cameras per room.
-- This cannot be expressed as a plain CHECK constraint in PostgreSQL because it
-- depends on counting rows in another table; use a constraint trigger instead.
CREATE OR REPLACE FUNCTION enforce_three_cameras_per_room()
RETURNS trigger AS $$
DECLARE
    cam_count integer;
    target_room uuid;
BEGIN
    target_room := COALESCE(NEW.room_id, OLD.room_id);

    SELECT COUNT(*) INTO cam_count
    FROM cameras
    WHERE room_id = target_room;

    IF cam_count <> 3 THEN
        RAISE EXCEPTION 'Room % must have exactly 3 cameras; found %', target_room, cam_count;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    CREATE CONSTRAINT TRIGGER cameras_exactly_three_per_room
    AFTER INSERT OR UPDATE OR DELETE ON cameras
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    EXECUTE FUNCTION enforce_three_cameras_per_room();
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

-- ── Students ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS students (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    section_id  UUID NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    roll_no     TEXT NOT NULL,
    name        TEXT NOT NULL,
    model_pack  TEXT NOT NULL,
    registered  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (section_id, roll_no)
);

-- One row PER embedding — never averaged.
-- 80-100 students × 5-8 embeddings ≈ 400-800 vectors per section.
-- Detection loads only the section's FAISS index (never searches college-wide).
CREATE TABLE IF NOT EXISTS student_embeddings (
    id           BIGSERIAL PRIMARY KEY,
    student_id   UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    section_id   UUID NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    embedding    vector(512) NOT NULL,
    landmarks    JSONB,                -- 5-point mask: [[x,y]*5]
    aligned_mask BYTEA,               -- 112x112 aligned crop (JPEG)
    angle_hint   TEXT,                -- 'front','left','top', etc.
    det_score    REAL,
    source       TEXT NOT NULL DEFAULT 'registration',  -- vs 'auto_enroll'
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Section-scoped queries are the hot path — index on section_id.
CREATE INDEX IF NOT EXISTS idx_emb_section ON student_embeddings (section_id);
CREATE INDEX IF NOT EXISTS idx_emb_student ON student_embeddings (student_id);
