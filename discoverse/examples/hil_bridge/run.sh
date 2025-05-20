#!/bin/bash
set -e

if [ -n "${MODEL_PATH}" ]; then
    MOUNT_ARGS="-v ${MODEL_PATH}:/workspace/models"
fi

docker run -it --rm \
    --network host \
    -e ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    --privileged -v /usr/lib/$(uname -m)-linux-gnu/:/usr/lib/$(uname -m)-linux-gnu/ \
    ${MOUNT_ARGS} \
    ${IMAGE_REF:-"registry.cn-shanghai.aliyuncs.com/discover-robotics/simulator:full"} \
    python3 ros/play.py $@