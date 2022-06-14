CREATE TABLE IF NOT EXISTS `event_metadata` (
  `event_id` VARCHAR(100) NOT NULL,
  `camera_info` JSON NOT NULL,
  PRIMARY KEY (`event_id`),
  CONSTRAINT `fk_event_metadata_event1`
    FOREIGN KEY (`event_id`)
    REFERENCES `event` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
