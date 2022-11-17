CREATE TABLE IF NOT EXISTS `event` (
  `id` VARCHAR(100) NOT NULL,
  `people_id` INT NOT NULL,
  `object_id` INT NOT NULL,
  `camera_id` INT NOT NULL,
  `pose_id` INT NOT NULL,
  `action` VARCHAR(45) NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  INDEX `fk_event_people1_idx` (`people_id` ASC) INVISIBLE,
  INDEX `fk_event_object1_idx` (`object_id` ASC) VISIBLE,
  INDEX `fk_event_camera1_idx` (`camera_id` ASC) VISIBLE,
  CONSTRAINT `fk_event_people1`
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
