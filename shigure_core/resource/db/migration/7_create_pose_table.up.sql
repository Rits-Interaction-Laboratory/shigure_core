CREATE TABLE IF NOT EXISTS `pose` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `sec` VARCHAR(100) NOT NULL,
  `nanosec` VARCHAR(1024) NOT NULL,
  `frame_id` INT NOT NULL,

  `nose_x` FLOAT NOT NULL,
  `nose_y` FLOAT NOT NULL,
  `nose_z` FLOAT NOT NULL,
  
  `neck_x` FLOAT NOT NULL,
  `neck_y` FLOAT NOT NULL,
  `neck_z` FLOAT NOT NULL,
  
  `right_shoulder_x` FLOAT NOT NULL,
  `right_shoulder_y` FLOAT NOT NULL,
  `right_shoulder_z` FLOAT NOT NULL,
  
  `right_elbow_x` FLOAT NOT NULL,
  `right_elbow_y` FLOAT NOT NULL,
  `right_elbow_z` FLOAT NOT NULL,
  
  `right_wrist_x` FLOAT NOT NULL,
  `right_wrist_y` FLOAT NOT NULL,
  `right_wrist_z` FLOAT NOT NULL,
  
  `left_shoulder_x` FLOAT NOT NULL,
  `left_shoulder_y` FLOAT NOT NULL,
  `left_shoulder_z` FLOAT NOT NULL,
  
  `left_elbow_x` FLOAT NOT NULL,
  `left_elbow_y` FLOAT NOT NULL,
  `left_elbow_z` FLOAT NOT NULL,
  
  `left_wrist_x` FLOAT NOT NULL,
  `left_wrist_y` FLOAT NOT NULL,
  `left_wrist_z` FLOAT NOT NULL,
  
  `mid_hip_x` FLOAT NOT NULL,
  `mid_hip_y` FLOAT NOT NULL,
  `mid_hip_z` FLOAT NOT NULL,
  
  `right_hip_x` FLOAT NOT NULL,
  `right_hip_y` FLOAT NOT NULL,
  `right_hip_z` FLOAT NOT NULL,
  
  `right_knee_x` FLOAT NOT NULL,
  `right_knee_y` FLOAT NOT NULL,
  `right_knee_z` FLOAT NOT NULL,
  
  `right_ankle_x` FLOAT NOT NULL,
  `right_ankle_y` FLOAT NOT NULL,
  `right_ankle_z` FLOAT NOT NULL,
  
  `left_hip_x` FLOAT NOT NULL,
  `left_hip_y` FLOAT NOT NULL,
  `left_hip_z` FLOAT NOT NULL,

  `left_knee_x` FLOAT NOT NULL,
  `left_knee_y` FLOAT NOT NULL,
  `left_knee_z` FLOAT NOT NULL,

  `left_ankle_x` FLOAT NOT NULL,
  `left_ankle_y` FLOAT NOT NULL,
  `left_ankle_z` FLOAT NOT NULL,

  `right_eye_x` FLOAT NOT NULL,
  `right_eye_y` FLOAT NOT NULL,
  `right_eye_z` FLOAT NOT NULL,

  `left_eye_x` FLOAT NOT NULL,
  `left_eye_y` FLOAT NOT NULL,
  `left_eye_z` FLOAT NOT NULL,

  `right_ear_x` FLOAT NOT NULL,
  `right_ear_y` FLOAT NOT NULL,
  `right_ear_z` FLOAT NOT NULL,

  `left_ear_x` FLOAT NOT NULL,
  `left_ear_y` FLOAT NOT NULL,
  `left_ear_z` FLOAT NOT NULL,

  `left_big_toe_x` FLOAT NOT NULL,
  `left_big_toe_y` FLOAT NOT NULL,
  `left_big_toe_z` FLOAT NOT NULL,

  `left_small_toe_x` FLOAT NOT NULL,
  `left_small_toe_y` FLOAT NOT NULL,
  `left_small_toe_z` FLOAT NOT NULL,
  
  `left_heel_x` FLOAT NOT NULL,
  `left_heel_y` FLOAT NOT NULL,
  `left_heel_z` FLOAT NOT NULL,

  `right_big_toe_x` FLOAT NOT NULL,
  `right_big_toe_y` FLOAT NOT NULL,
  `right_big_toe_z` FLOAT NOT NULL,

  `right_small_toe_x` FLOAT NOT NULL,
  `right_small_toe_y` FLOAT NOT NULL,
  `right_small_toe_z` FLOAT NOT NULL,

  `right_heel_x` FLOAT NOT NULL,
  `right_heel_y` FLOAT NOT NULL,
  `right_heel_z` FLOAT NOT NULL,

  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));