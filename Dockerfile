FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    xterm \
    ros-humble-rviz2 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install "numpy<2" mysql-connector-python pandas

WORKDIR /ros2_ws/src
COPY bbox_ex_msgs/ bbox_ex_msgs/
COPY shigure_core_msgs/ shigure_core_msgs/
COPY shigure_core/ shigure_core/
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
CMD ["ros2", "launch", "shigure_core", "shigure_core_docker_launch.py"]
