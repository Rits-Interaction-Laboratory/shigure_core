# Shigure Core
![Shigure Core](https://img.shields.io/badge/shigure-core-red)
室内監視システム ROS2版

## Requires
* ROS2 Dashing Diademata [公式インストール方法](https://index.ros.org/doc/ros2/Installation/Dashing/)
* Intel® RealSense™ D435
* ROS2 Wrapper for Intel® RealSense™ Devices [公式リポジトリ](https://github.com/intel/ros2_intel_realsense)
* OpenCV
* OpenPose
* (OPTIONAL) web_video_server [ROS wiki](https://wiki.ros.org/web_video_server) [インストール方法](https://github.com/RobotWebTools/web_video_server/issues/108)

## PacageName
* Shigure

## Nodes
* bg_subtraction_node
* bg_subtraction_preview_node

## Topics
### bg_subtraction_node
* /shigure/bg_subtraction
    * 本番用の背景差分用topic
### bg_subtraction_preview_node
* /shigure/preview/bg_subtraction/depth
    * 深度画像のカラーマップ画像
* /shigure/preview/bg_subtraction/avg
    * 深度の平均画像のカラーマップ画像
* /shigure/preview/bg_subtraction/sd
    * 深度の標準偏差のカラーマップ画像
* /shigure/preview/bg_subtraction/result
    * 背景差分のデバッグ画像
    * 画像にframe数, fpsが書き込まれている 

## EntoryPoints
* bg_subtraction
    * 本番用背景差分ノード
* bg_subtraction_preview
    * デバッグ用背景差分ノード

## 起動方法
1. `ros2 run realsense_ros2_camera realsense_ros2_camera` でRealSenseを起動する
1. `ros2 run shigure %{EntoryPointName}` で任意のノードを起動する
    * 例： `ros2 run shigure bg_subtraction` で本番用背景差分ノードを起動
