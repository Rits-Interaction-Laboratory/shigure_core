ALTER TABLE `object_history`
  DROP INDEX `idx_object_history_bring_in_at`,
  DROP INDEX `idx_object_history_take_out_at`,
  DROP INDEX `idx_object_history_last_seen_at`;

ALTER TABLE `object_tracking_summary`
  DROP INDEX `idx_object_tracking_summary_bring_in_at`,
  DROP INDEX `idx_object_tracking_summary_take_out_at`;
