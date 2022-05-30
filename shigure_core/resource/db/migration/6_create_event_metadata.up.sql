CREATE TABLE IF NOT EXISTS `event_metadata` (
  `event_id` INT NOT NULL,
  `width` INT NOT NULL,
  `height` INT NOT NULL,
  `x` INT NOT NULL,
  `y` INT NOT NULL,
  `z` INT NOT NULL,
  PRIMARY KEY (`event_id`),
  CONSTRAINT `fk_event_metadata_event1`
    FOREIGN KEY (`event_id`)
    REFERENCES `event` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
