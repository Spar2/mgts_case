"""Ordered, additive control-plane migrations executed under the schema lock."""

MIGRATIONS = {
    3: "ALTER TABLE jobs ADD COLUMN submitted_by text REFERENCES users; CREATE INDEX jobs_user_active ON jobs(project_id,submitted_by,status);",
    2: """
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created double precision NOT NULL DEFAULT extract(epoch from now());
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_seen double precision NOT NULL DEFAULT extract(epoch from now());
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS label text NOT NULL DEFAULT 'Browser';
CREATE TABLE invitations (
 id text PRIMARY KEY, digest text UNIQUE NOT NULL, project_id text NOT NULL REFERENCES projects,
 role text NOT NULL CHECK(role IN ('viewer','expert','analyst','admin')),
 created_by text REFERENCES users, created double precision NOT NULL,
 expires double precision NOT NULL, consumed double precision, revoked boolean NOT NULL DEFAULT false
);
CREATE TABLE password_resets (
 id text PRIMARY KEY, digest text UNIQUE NOT NULL, user_id text NOT NULL REFERENCES users,
 created_by text REFERENCES users, created double precision NOT NULL,
 expires double precision NOT NULL, consumed double precision, revoked boolean NOT NULL DEFAULT false
);
CREATE TABLE access_requests (
 id text PRIMARY KEY, name text NOT NULL, contact text NOT NULL, reason text NOT NULL,
 created double precision NOT NULL, closed double precision,
 status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','declined'))
);
CREATE INDEX sessions_user ON sessions(user_id);
CREATE INDEX access_requests_retention ON access_requests(closed);
""",
}


def migrate(db):
    current = db.execute("SELECT max(version) FROM platform_schema").fetchone()[0]
    if current > max(MIGRATIONS):
        raise ValueError("Database schema is newer than this application")
    for version, sql in sorted(MIGRATIONS.items()):
        if version > current:
            db.execute(sql)
            db.execute("INSERT INTO platform_schema VALUES(%s)", (version,))
