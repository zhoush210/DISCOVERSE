# DISCOVERSE Troubleshooting Guide

This guide helps you resolve common issues when installing and using DISCOVERSE. Issues are organized by category for easy navigation.

## Table of Contents

- [Installation Issues](#installation-issues)
  - [CUDA and PyTorch](#cuda-and-pytorch)
  - [Dependencies](#dependencies)
  - [Submodules](#submodules)
- [Runtime Issues](#runtime-issues)
  - [Graphics and Display](#graphics-and-display)
  - [Server Deployment](#server-deployment)

---

## Installation Issues

### CUDA and PyTorch

#### CUDA/PyTorch Version Mismatch

**Problem**: `diff-gaussian-rasterization` fails to install with error message about mismatched PyTorch and CUDA versions.

**Solution**: Install matching PyTorch version for your CUDA installation:

```bash
# For CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

> **Tip**: Check your CUDA version with `nvcc --version` or `nvidia-smi`

#### Missing GLM Headers

**Problem**: Compilation error with missing `glm/glm.hpp` header file.

```
fatal error: glm/glm.hpp: no such file or directory
```

**Solution**: Install GLM library and update include path:

```bash
# Using conda (recommended)
conda install -c conda-forge glm
export CPATH=$CONDA_PREFIX/include:$CPATH

# Then reinstall diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization
```

### Dependencies

#### Taichi Installation Failure

**Problem**: Taichi fails to install or import properly.

**Solution**: Install specific Taichi version:

```bash
pip install taichi==1.6.0
```

#### PyQt5 Installation Issues

**Problem**: PyQt5 installation fails or GUI components don't work.

**Solution**: Install system packages first:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pyqt5 python3-pyqt5-dev

# Then install via pip
pip install PyQt5>=5.15.0
```

### Submodules

#### Submodules Not Initialized

**Problem**: Missing submodule content or import errors from submodules.

**Solution**: Initialize submodules using one of these methods:

```bash
# Method 1: On-demand initialization (recommended)
python setup_submodules.py --list              # Check status
python setup_submodules.py --module lidar act  # Initialize specific modules
python setup_submodules.py --all               # Initialize all modules

# Method 2: Traditional Git approach
git submodule update --init --recursive
```

---

## Runtime Issues

### Graphics and Display

Graphics rendering issues in DISCOVERSE typically fall into three categories, each with different root causes and solutions.

#### GLX Configuration Errors

**Problem**: GLFW/OpenGL initialization fails with GLX errors:

```
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
```

**Root Cause**: X11/GLX configuration issues, often due to:
- Dual GPU systems (Intel + NVIDIA) with driver conflicts
- Missing or misconfigured X11 display server
- Incompatible GLX extensions

**Solutions**:

1. **For systems with NVIDIA GPU**: Check and configure graphics driver mode (dual GPU systems):
   ```bash
   # Check current driver mode
   prime-select query
   
   # If output is 'on-demand', switch to NVIDIA mode
   sudo prime-select nvidia

   # Force NVIDIA usage
   export __NV_PRIME_RENDER_OFFLOAD=1
   export __GLX_VENDOR_LIBRARY_NAME=nvidia

   # Reboot system after switching
   sudo reboot
   ```

2. **For systems without NVIDIA GPU** (conda environments):
   
   **Root Cause**: Low version of libstdc++ in conda environment causing GLX compatibility issues.
   
   **Solution 1** - Install newer libstdc++ in conda environment:
   ```bash
   conda install -c conda-forge libstdcxx-ng
   ```

   **Solution 2** - Use system libstdc++ library:
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
   ```
   
3. **Verify GLX support**:
   ```bash
   glxinfo | grep "direct rendering"
   glxgears  # Test basic GLX functionality
   ```

5. **For X11 display issues**:
   ```bash
   # Ensure X11 forwarding (if using SSH)
   ssh -X username@hostname
   
   # Check DISPLAY variable
   echo $DISPLAY
   ```

#### EGL Initialization Errors

**Problem**: EGL backend fails to initialize, especially in virtual/containerized environments:

```
libEGL warning: MESA-LOADER: failed to open virtio_gpu: /usr/lib/dri/virtio_gpu_dri.so: cannot open shared object file
libEGL warning: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file
GLFWError: (65542) b'EGL: Failed to initialize EGL: EGL is not or could not be initialized'
```

**Root Cause**: Missing or incompatible Mesa drivers, particularly in:
- Virtual machines (VirtIO GPU driver issues)
- Docker containers without proper GPU passthrough
- ARM-based systems with incomplete driver installations

**Solutions**:

1. **Install Mesa drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install mesa-utils libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev
   
   # For virtual environments, also install
   sudo apt-get install mesa-vulkan-drivers mesa-va-drivers
   ```

2. **For VirtIO GPU issues**:
   ```bash
   # Install VirtIO GPU drivers
   sudo apt-get install xserver-xorg-video-qxl
   
   # Or fall back to software rendering
   export LIBGL_ALWAYS_SOFTWARE=1
   ```

3. **Configure EGL for headless rendering**:
   ```bash
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   ```

#### MuJoCo-Specific Rendering Issues

**Problem**: MuJoCo environments fail to render properly despite working graphics drivers.

**Root Cause**: MuJoCo's specific rendering backend requirements and conflicts with system OpenGL configurations.

**Solutions**:

1. **Set MuJoCo rendering backend**:
   ```bash
   # For headless servers
   export MUJOCO_GL=egl
   
   # For desktop environments with display issues
   export MUJOCO_GL=glfw
   
   # For software rendering (fallback)
   export MUJOCO_GL=osmesa
   ```

2. **Verify MuJoCo installation**:
   ```bash
   python -c "import mujoco; mujoco.MjModel.from_xml_string('<mujoco/>')"
   ```

3. **Test with simple MuJoCo example**:
   ```python
   import mujoco
   import mujoco.viewer
   
   # Simple test model
   xml = """
   <mujoco>
     <worldbody>
       <geom name="floor" type="plane" size="0 0 .05"/>
       <body name="box" pos="0 0 .2">
         <geom name="box" type="box" size=".1 .1 .1"/>
       </body>
     </worldbody>
   </mujoco>
   """
   
   model = mujoco.MjModel.from_xml_string(xml)
   data = mujoco.MjData(model)
   
   # Test rendering
   with mujoco.viewer.launch_passive(model, data) as viewer:
       mujoco.mj_step(model, data)
   ```

> **Reference**: Similar issues reported in [Gymnasium Issue #755](https://github.com/Farama-Foundation/Gymnasium/issues/755#issuecomment-2825928509)

### Server Deployment

#### Headless Server Setup

**Problem**: Running DISCOVERSE on a server without display.

**Solution**: Configure MuJoCo for headless rendering:

```bash
export MUJOCO_GL=egl
```

Add this to your shell profile (`.bashrc`, `.zshrc`) for permanent effect:

```bash
echo "export MUJOCO_GL=egl" >> ~/.bashrc
source ~/.bashrc
```

---

## Getting Help

If your issue isn't covered here:

1. **Search GitHub Issues**: Check [existing issues](https://github.com/TATP-233/DISCOVERSE/issues) for similar problems
2. **Create New Issue**: Provide detailed error messages and system information
3. **Community Support**: Join our WeChat community for real-time help
4. **Documentation**: Check the `/doc` directory for detailed guides

### Issue Report Template

When reporting issues, please include:

```
**System Information:**
- OS: (e.g., Ubuntu 22.04)
- Python version: 
- CUDA version: 
- GPU model: 

**Error Message:**
[Paste complete error trace here]

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**
[What should happen]

**Additional Context:**
[Any other relevant information]
```

---

> **Note**: This troubleshooting guide is actively maintained. If you find a solution to a problem not listed here, please consider contributing to help other users. 