ALTER TABLE `object_history`
  ADD INDEX `idx_object_history_bring_in_at` (`bring_in_at`),
  ADD INDEX `idx_object_history_take_out_at` (`take_out_at`),
  ADD INDEX `idx_object_history_last_seen_at` (`last_seen_at`);

ALTER TABLE `object_tracking_summary`
  ADD INDEX `idx_object_tracking_summary_bring_in_at` (`bring_in_at`),
  ADD INDEX `idx_object_tracking_summary_take_out_at` (`take_out_at`);
