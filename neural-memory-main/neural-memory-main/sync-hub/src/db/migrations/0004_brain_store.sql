-- Brain Store: ratings stored in D1, brain packages stored on GitHub.
-- Migration 0004

-- Ratings only — brain packages live on GitHub (unlimited free storage).
CREATE TABLE IF NOT EXISTS brain_ratings (
  id TEXT PRIMARY KEY,
  brain_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
  comment TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(brain_name, user_id)
);

CREATE INDEX IF NOT EXISTS idx_br_brain_name ON brain_ratings(brain_name);
CREATE INDEX IF NOT EXISTS idx_br_user_id ON brain_ratings(user_id);
