ARG ROS_VERSION=foxy
FROM ros:$ROS_VERSION

SHELL ["/bin/bash", "-c"]

ARG ROS_VERSION=foxy
RUN sudo apt-get update && sudo apt-get install -y \
    git \
    nano \
    vim \
    python3-pip \
    libeigen3-dev \
    tmux \
    ros-${ROS_VERSION}-rqt-graph \
    ros-${ROS_VERSION}-ackermann-msgs \
    ros-${ROS_VERSION}-rviz2
RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d

ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]