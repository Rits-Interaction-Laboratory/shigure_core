CREATE TABLE IF NOT EXISTS `event` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `people_id` VARCHAR(100) NOT NULL,
  `object_id` VARCHAR(100) NOT NULL,
  `camera_id` INT NOT NULL,
  `action` VARCHAR(45) NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  INDEX `fk_event_people_idx` (`people_id` ASC) VISIBLE,
  INDEX `fk_event_object1_idx` (`object_id` ASC) VISIBLE,
  INDEX `fk_event_camera1_idx` (`camera_id` ASC) VISIBLE,
  CONSTRAINT `fk_event_people`
    FOREIGN KEY (`people_id`)
    REFERENCES `people` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_event_object1`
    FOREIGN KEY (`object_id`)
    REFERENCES `object` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_event_camera1`
    FOREIGN KEY (`camera_id`)
    REFERENCES `camera` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
