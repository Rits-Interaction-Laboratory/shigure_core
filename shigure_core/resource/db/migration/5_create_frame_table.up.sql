CREATE TABLE IF NOT EXISTS `frame` (
  `event_id` VARCHAR(100) NOT NULL,
  `frame_count` INT NOT NULL,
  `color_path` VARCHAR(1024) NOT NULL,
  `depth_path` VARCHAR(1024) NOT NULL,
  `points_path` VARCHAR(1024) NOT NULL,
  INDEX `fk_frame_event1_idx` (`event_id` ASC) VISIBLE,
  PRIMARY KEY (`event_id`, `frame_count`),
  CONSTRAINT `fk_frame_event1`
    FOREIGN KEY (`event_id`)
    REFERENCES `event` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
