# Minimized Simulator Docker with ROS

This is a minimized version for simulator without 3DGS rendering and with ROS runtime.

## Build

The prebuilt images can be downloaded from 

```shell
# Runtime version, the models directory should be mounted 
docker pull registry.cn-shanghai.aliyuncs.com/discover-robotics/simulator:runtime
# Full version, the models directory with basic model files is already included
docker pull registry.cn-shanghai.aliyuncs.com/discover-robotics/simulator:full
```

To locally build the runtime image (without model files):

```shell
docker build -f ros/Dockerfile.ros --target sim-runtime . -t ${IMAGE_NAME:-"sim-runtime"}
```

To locally build the full image (with model files, provided that directory `models` exists under docker build context directory):

```shell
docker build -f ros/Dockerfile.ros --target sim-full . -t ${IMAGE_NAME:-"sim-full"}
```

## Use

To launch the simulator with runtime image:

```shell
IMAGE_REF=sim-runtime MODEL_PATH=./models ./ros/run.sh
```

To launch the simulator with full image:

```shell
IMAGE_REF=sim-full ./ros/run.sh
```