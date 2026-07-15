FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    build-essential \
    python3-dev \
    xterm \
    ros-humble-rviz2 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Python依存はルートのrequirements.txtで一元管理（生環境と共通）
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# insightface のモデル(buffalo_s)をビルド時にダウンロードしておく
# (初回起動時のダウンロード待ち・実行時のネットワーク依存をなくすため)
RUN python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])"

WORKDIR /ros2_ws/src
COPY bbox_ex_msgs/ bbox_ex_msgs/
COPY shigure_core_msgs/ shigure_core_msgs/
COPY shigure_core/ shigure_core/
COPY shigure_api/ shigure_api/
RUN git clone https://github.com/Rits-Interaction-Laboratory/openpose_ros2 \
    && rm -rf openpose_ros2/openpose_ros2
RUN git clone https://github.com/Rits-Interaction-Laboratory/people_detection_ros2 \
    && rm -rf people_detection_ros2/people_detection_ros2

WORKDIR /ros2_ws
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

RUN sed -i '/^exec/i source /ros2_ws/install/setup.bash' /ros_entrypoint.sh

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "shigure_core", "shigure_core_launch.py", "terminal:=xterm", "record:=true", "save_root_path:=/ros2_ws/events", "debug_mode:=true"]
