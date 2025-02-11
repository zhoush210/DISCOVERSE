## Build server

```bash
cd DISCOVERSE/discoverse/examples/s2r2025/docker
docker build -f Dockerfile.server -t discoverse:s2r_server <PATH-TO-DISCOVERSE>

cd DISCOVERSE
docker run -dit --rm \
    --name s2r_server \
    --gpus all \
    --privileged=true \
    --network=host \
    --ipc=host \
    --pid=host \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=0 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/input:/dev/input \
    -v $(pwd):/workspace \
    discoverse:s2r_server bash
```

## Build client

```bash
cd DISCOVERSE/discoverse/examples/s2r2025/docker
docker build -f Dockerfile.client -t <YOUR-TEAM-NAME>:<TAG> .

docker run -dit --rm \
    --network=host \
    --ipc=host \
    --pid=host \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=0 \
    <YOUR-TEAM-NAME>:<TAG> bash
```

## 测试通讯

```bash
# # server容器中发布通信测试
source /opt/ros/humble/setup.bash
ros2 topic pub /test1 std_msgs/msg/String "data: 'hello from server'"

# # client容器中发布通信测试
source /opt/ros/humble/setup.bash
ros2 topic pub /test2 std_msgs/msg/String "data: 'hello from client'"

# client/server 容器 查看所有活动的topics
ros2 topic list

# 查看详细信息
ros2 topic info /test1
ros2 topic info /test2

# 查看topic类型
# ros2 interface show std_msgs/msg/String

```