/**
* savedata : 記録開始→記録終了までの1かたまりごとの番号
* save_id : savedata内での順序を決めるid
**/
CREATE TABLE IF NOT EXISTS `pose` (
  `id` INT NOT NULL AUTO_INCREMENT,
  'savedata' INT NOT NULL,
  'save_id' INT NOT NULL, 
  `event_id` VARCHAR(100),
  `pose_key_points_list` JSON NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));