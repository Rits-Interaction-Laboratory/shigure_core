CREATE TABLE IF NOT EXISTS `pose` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `savedata` INT NOT NULL,
  `save_id` INT NOT NULL, 
  `pose_key_points_list` JSON NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));