CREATE TABLE IF NOT EXISTS `pose` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `sequence_id` INT NOT NULL,
  `frame_number` INT NOT NULL, 
  `pose_key_points_list` JSON NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));