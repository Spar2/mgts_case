-- Additive migration: old tables and original data are preserved.
CREATE TABLE IF NOT EXISTS haklolarp.profiles_v2
(
 replay_id String,
 source_sha256 String,
 dts_id String,
 channel_id String,
 ts_raw String,
 ts Nullable(DateTime64(6,'UTC')),
 time_kind LowCardinality(String),
 elapsed_s Float64,
 row_idx UInt32,
 segment_id UInt32,
 x_m Array(Float64),
 temp_c Array(Nullable(Float64)),
 invalid_reason Array(UInt8),
 payload_sha256 String,
 received_at DateTime64(6,'UTC') DEFAULT now64(6)
)
ENGINE=ReplacingMergeTree(received_at)
ORDER BY(replay_id,row_idx);

CREATE TABLE IF NOT EXISTS haklolarp.profile_conflicts
(
 replay_id String,row_idx UInt32,payload_sha256 String,reason String,at DateTime DEFAULT now()
)
ENGINE=MergeTree ORDER BY(replay_id,row_idx);
