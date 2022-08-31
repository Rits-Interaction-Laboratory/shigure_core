# Shigure Core
![Shigure Core](https://img.shields.io/badge/shigure-core-red)

ROS2による室内シーン変遷ロギングシステム 

Wiki : https://github.com/Rits-Interaction-Laboratory/shigure_core/wiki

## Requires
* ROS2 Foxy [公式インストール方法](https://index.ros.org/doc/ros2/Installation/Foxy/)
* Intel® RealSense™ D435
* ROS2 Wrapper for Intel® RealSense™ Devices [公式リポジトリ](https://github.com/intel/ros2_intel_realsense)
    * Rits-Interaction-Laboratory/rs_ros2_python [リポジトリ](https://github.com/Rits-Interaction-Laboratory/rs_ros2_python)
* Rits-Interaction-Laboratory/openpose_ros2 [リポジトリ](https://github.com/Rits-Interaction-Laboratory/openpose_ros2)
    * docker版 : Rits-Interaction-Laboratory/openpose_ros2_docker [リポジトリ](https://github.com/Rits-Interaction-Laboratory/openpose_ros2_docker)
* Rits-Interaction-Laboratory/people_detection_ros2 [リポジトリ](https://github.com/Rits-Interaction-Laboratory/people_detection_ros2)
    * docker版 : Rits-Interaction-Laboratory/people_detection_ros2_docker [リポジトリ](https://github.com/Rits-Interaction-Laboratory/people_detection_ros2)
* OpenCV
* Docker
* Docker Compose
* (OPTIONAL) web_video_server [ROS wiki](https://wiki.ros.org/web_video_server) [インストール方法](https://github.com/RobotWebTools/web_video_server/issues/108)


## 起動方法

### 1. RealSense nodeの起動

Rits-Interaction-Laboratory/rs_ros2_python を利用

```sh
ros2 run rs_ros2_python rs_camera
```

### 2. Openpose ROS2 nodeの起動

Rits-Interaction-Laboratory/openpose_ros2_dockerを利用

```sh
docker run -it --gpus all --net host openpose_ros2_docker
bash /run.bash
```

### 3. People Detection ROS2 nodeの起動

Rits-Interaction-Laboratory/people_detection_ros2_dockerを利用

```sh
docker run -it --gpus all --net host people_detection_ros2_docker
bash /run.bash
```

### 4. shigure_core(本リポジトリ)の起動

ビルド
```sh
colcon build
. <ROS2 workspace>/install/setup.bash
```

実行(全ノードを実行)
```sh
ros2 launch shigure_core shigure_core_launch.py
```

### 5. DBへの保存を行う場合

DBの起動 <br>
```sh
cd shigure_core/resource/db
docker-compose up
```

record_event nodeの起動 <br>
(パラメータ使用のためYMLファイルを読み込む。 <br>
Sample YMLファイルの配置フォルダ : [/shigure_core/shigure_core/shigure_core/nodes/params/](/shigure_core/shigure_core/shigure_core/nodes/params/))
```sh
ros2 run shigure_core record_event --ros-args --params-file <ROS2 workspace>/src/shigure_core/shigure_core/shigure_core/nodes/params/record_event_params.yml
```

DBへの接続 <br>
(PW : `shigure`) <br>
```sh
mysql -h 127.0.0.1 -P 3306 -u shigure -p
```
