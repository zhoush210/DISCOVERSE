# Domain Randomization for Sim-to-Real Transfer

## 1. Introduction to Domain Randomization

### 1.1 What is Domain Randomization?
Domain Randomization (DR) is a technique used primarily in robotics and reinforcement learning to improve the transfer of policies learned in simulation (sim) to the real world (real). The core idea is to train models on a wide distribution of simulated environments that vary significantly in their visual and physical properties. By exposing the model to a multitude of variations during training, it learns to be robust to the differences between simulation and reality, thereby bridging the "sim-to-real gap."

Variations in DR can include:
- **Visual Domain**: Lighting conditions, textures, colors, camera positions, distractors, etc.
- **Dynamics Domain**: Physical parameters like mass, friction, motor torques, sensor noise, etc.

### 1.2 Why Use Domain Randomization?
- **Bridging the Sim-to-Real Gap**: Simulations are imperfect approximations of the real world. DR helps models generalize better by not overfitting to a single, specific simulated environment.
- **Reducing Manual Effort**: Creating highly realistic simulations is time-consuming and expensive. DR allows for effective learning even with less photo-realistic, but more varied, simulations.
- **Data Augmentation**: DR acts as a powerful form of data augmentation, especially for deep learning models that require large amounts of diverse training data.
- **Improved Robustness**: Policies trained with DR are generally more robust to unexpected changes or noise in the real world.

## 2. The `discoverse/randomain` Tool

The `discoverse/randomain` toolkit, located under `discoverse/randomain`, provides a pipeline to apply domain randomization to video sequences, typically captured from a robotic simulation environment. It focuses on **visual domain randomization** by leveraging powerful generative models to alter the appearance of scenes frame by frame.

**Key functionalities include:**
- **Data Sampling**: Capturing synchronized RGB, depth, and segmentation mask information from a simulation.
- **Generative Scene Modification**: Using text prompts and depth information with state-of-the-art generative models (via ComfyUI) to re-render video frames with varied appearances.
- **Optical Flow for Temporal Consistency**: Employing optical flow techniques to generate intermediate frames, ensuring smoother transitions and reducing computational load.

The goal is to take a "clean" simulated video and produce multiple visually distinct versions of it, which can then be used to train more robust perception models or policies. All generated data and intermediate samples are typically stored in `data/randomain`.

## 3. Core Technologies

The `discoverse/randomain` pipeline integrates several advanced technologies:

### 3.1 Generative Models via ComfyUI

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a powerful and modular node-based graphical user interface for Stable Diffusion. This toolkit utilizes ComfyUI as a backend to drive generative models.

- **Stable Diffusion XL (SDXL) Turbo**:
    - **Principle**: SDXL Turbo is a distilled version of SDXL, designed for rapid, high-quality image generation from text prompts. It achieves this speed by reducing the number of diffusion steps typically required. The `sd_xl_turbo_1.0_fp16.safetensors` model is a 16-bit floating-point version, offering a balance between performance and precision.
    - **Role**: It serves as the primary text-to-image generation engine, creating novel visual appearances based on input prompts.

- **ControlNet (for Depth)**:
    - **Principle**: ControlNets are neural network structures that add extra conditions to pre-trained diffusion models like SDXL. The `controlnet_depth_sdxl_1.0` model is specifically trained to condition image generation on depth maps. This means it can generate an image that is consistent with a given 3D scene structure (as defined by the depth map) while varying the texture, style, and objects based on the text prompt.
    - **Role**: Crucial for maintaining the geometric integrity of the original scene. By using the depth map from the simulation, ControlNet ensures that the randomized scene elements respect the original object placements, shapes, and overall 3D layout.

- **VAE (Variational Autoencoder)**:
    - **Principle**: VAEs are used in latent diffusion models (like Stable Diffusion) to encode images into a lower-dimensional latent representation and decode them back into pixel space. The `sdxl_vae.safetensors` model is optimized for SDXL.
    - **Role**: Handles the compression and decompression of image data, enabling the diffusion process to occur in a more manageable latent space, which is computationally cheaper.

### 3.2 Optical Flow

Optical flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer (e.g., a camera) and the scene.

- **Purpose in `randomain`**: When generating a sequence of randomized frames, running the full generative model for every single frame can be computationally expensive. If a `flow_interval` greater than 1 is specified, the generative model is run only on keyframes. The frames *between* these keyframes are then interpolated using optical flow. This speeds up the overall process while attempting to maintain temporal smoothness.

- **Supported Methods**:
    - **Farneback Method (`rgb`)**:
        - **Principle**: A classical algorithm that computes dense optical flow (i.e., flow vectors for every pixel) based on Gunnar Farneback's polynomial expansion. It analyzes image intensity changes between two consecutive frames.
        - **Pros**: Relatively lightweight, doesn't require a GPU or pre-trained model.
        - **Cons**: Can be less accurate than deep learning methods, especially with large displacements, occlusions, or complex textures.
    - **RAFT (Recurrent All-Pairs Field Transforms) (`raft`)**:
        - **Principle**: A state-of-the-art deep learning model for optical flow. It uses a recurrent architecture to iteratively update a flow field by looking up features in a correlation volume constructed from all pairs of pixels.
        - **Pros**: Generally more accurate and robust, especially for complex scenarios and larger motions.
        - **Cons**: Requires a pre-trained model (weights provided via Google Drive link) and is computationally more intensive, often benefiting from GPU acceleration.

## 4. Setup and Installation

### 4.1 Generative Model Setup (ComfyUI)

1.  **Clone ComfyUI**:
    ```bash
    # Navigate to your preferred directory for submodules or external tools
    # For example, if your project root is DISCOVERSE:
    # mkdir -p submodules
    # cd submodules
    git clone https://github.com/comfyanonymous/ComfyUI
    ```

2.  **Install ComfyUI Dependencies**:
    ```bash
    cd ComfyUI
    pip install -r requirements.txt
    cd ../.. # Return to project root or relevant directory
    ```

### 4.2 Model Deployment

The `randomain` tool expects specific generative models to be placed in a particular directory structure. Assuming your ComfyUI is at `submodules/ComfyUI`, the models should be placed within `submodules/ComfyUI/models`.

- **Checkpoints (Main Model)**:
    - Model: `sd_xl_turbo_1.0_fp16.safetensors`
    - Download from: [Hugging Face - stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors)
    - Deployment Path: `submodules/ComfyUI/models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors`

- **ControlNet (Depth Model)**:
    - Model: `controlnet_depth_sdxl_1.0.safetensors` (or `diffusion_pytorch_model.safetensors` renamed)
    - Download from: [Hugging Face - diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/blob/main/diffusion_pytorch_model.safetensors) (You might need to rename this file to match `controlnet_depth_sdxl_1.0.safetensors` if the script expects that exact name, or adjust paths in ComfyUI workflows if applicable). The example shows `controlnet_depth_sdxl_1.0.safetensors`.
    - Deployment Path: `submodules/ComfyUI/models/controlnet/controlnet_depth_sdxl_1.0.safetensors`

- **VAE (Variational Autoencoder)**:
    - Model: `sdxl_vae.safetensors` (or `diffusion_pytorch_model.safetensors` renamed)
    - Download from: [Hugging Face - stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae/blob/main/diffusion_pytorch_model.safetensors) (Similar to ControlNet, you might need to rename this file to `sdxl_vae.safetensors`).
    - Deployment Path: `submodules/ComfyUI/models/vae/sdxl_vae.safetensors`

**Directory Structure inside `submodules/ComfyUI` should look like:**
```
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   └── sd_xl_turbo_1.0_fp16.safetensors
│   ├── controlnet/
│   │   └── controlnet_depth_sdxl_1.0.safetensors
│   └── vae/
│       └── sdxl_vae.safetensors
├── ... (other ComfyUI files and folders)
```
The original document mentions an `extra_model_paths.yaml` within a `models` directory at the root of `randomain`. This seems to be a ComfyUI feature to specify alternative model locations. If you are using this `extra_model_paths.yaml` (e.g. `discoverse/randomain/models/extra_model_paths.yaml`), ensure it correctly points to wherever you've stored your models if not in the default ComfyUI paths. The example from the original doc:
```
randomain/ # This seems to refer to discoverse/randomain
├── models/
│   ├── checkpoints/
│   │   └── sd_xl_turbo_1.0_fp16.safetensors
│   ├── controlnet/
│   │   └── controlnet_depth_sdxl_1.0.safetensors
│   ├── extra_model_paths.yaml  # ComfyUI looks for this
│   └── vae/
│       └── sdxl_vae.safetensors
```
If using this structure, `extra_model_paths.yaml` might contain paths like:
```yaml
# Example content for extra_model_paths.yaml
# This tells ComfyUI to look in these specific subdirectories relative to this yaml's location or an absolute path.
checkpoints: ./checkpoints
controlnet: ./controlnet
vae: ./vae
# Or, if models are elsewhere:
# checkpoints: /path/to/my/global/checkpoints
```
The key is that ComfyUI must be able to find these models.

### 4.3 Environment Configuration (Path Linking)

For the `discoverse/randomain` scripts to correctly interface with ComfyUI and its models, you need to set up environment variables:

1.  **`PYTHONPATH`**: Add the path to the ComfyUI directory so Python can import its modules.
    ```bash
    export PYTHONPATH=/path/to/your/ComfyUI:$PYTHONPATH
    # Example if ComfyUI is in submodules/ComfyUI:
    # export PYTHONPATH=$(pwd)/submodules/ComfyUI:$PYTHONPATH 
    # (Ensure you run this from your project root or use an absolute path)
    ```

2.  **`COMFYUI_CONFIG_PATH`**: If you are using an `extra_model_paths.yaml` to tell ComfyUI where your models are (especially if they are not in the default `ComfyUI/models` subdirectories), you need to point ComfyUI to this configuration file.
    ```bash
    export COMFYUI_CONFIG_PATH=/path/to/your/randomain/models/extra_model_paths.yaml
    # Example if it's in discoverse/randomain/models/extra_model_paths.yaml:
    # export COMFYUI_CONFIG_PATH=$(pwd)/discoverse/randomain/models/extra_model_paths.yaml
    ```
    If you place all models directly into the standard `ComfyUI/models/...` subdirectories, you might not strictly need `extra_model_paths.yaml` or `COMFYUI_CONFIG_PATH`, as ComfyUI checks default locations. However, the original document implies its use.

**Recommendation**: Add these `export` commands to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`) or to a specific environment activation script for your project to avoid setting them manually each time.

### 4.4 Optical Flow Model Setup (RAFT)

-   If you choose to use the `RAFT` method for optical flow (`flow_method='raft'`):
    -   Download the pre-trained RAFT weights. The original document points to a Google Drive folder: [RAFT Models](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT). You'll likely need the `raft-things.pth` (for general optical flow) or a similar model.
    -   Place the downloaded `.pth` file into `discoverse/randomain/models/flow/`. For example:
        ```
        discoverse/randomain/
        ├── models/
        │   ├── flow/
        │   │   └── raft-things.pth # Or the specific model name you download
        ```
-   If using the `Farneback` method (`flow_method='rgb'`), no extra model download is needed as it's an OpenCV algorithm.
-   To integrate other optical flow methods, you would need to implement a similar interface to `FlowCompute/raft` within the `discoverse/randomain` codebase.

## 5. Usage Workflow

The process of generating randomized data involves three main stages:

### 5.1 Stage 1: Sample Collection

This stage involves running your existing simulation (e.g., a robotic arm performing a task) and capturing the necessary visual data frame-by-frame. The `discoverse.randomain.utils.SampleforDR` class is designed for this.

1.  **Instantiate `SampleforDR`**:
    ```python
    from discoverse.randomain.utils import SampleforDR

    # Example configuration (cfg would come from your project's config system)
    # objs = ['block_green', 'bowl_pink'] # List of key manipulable objects
    # robot_parts = ['panda_link0', 'panda_link1', ..., 'panda_hand'] # List of robot link names for segmentation
    # cam_ids = cfg.obs_rgb_cam_id # List or single ID of cameras to record from
    # save_dir = "data/randomain/trajectory_000" # Base directory for this trajectory's samples
    # fps = 30 
    # max_vis_dis = 1.0 # Maximum visualization distance for depth normalization (meters)

    samples = SampleforDR(
        objs=objs,
        robot_parts=robot_parts,
        cam_ids=cam_ids,
        save_dir=save_dir,
        fps=cfg.render_set.get('fps', fps), # Get FPS from config or use default
        max_vis_dis=max_vis_dis
    )
    ```
    -   **`objs`**: A list of strings, where each string is the unique name of a key dynamic object in the scene (e.g., manipulable blocks, tools, targets). These will have their individual segmentation masks saved.
    -   **`robot_parts`**: A list of strings, representing the names of the robot's links/parts. These will be grouped into a single "robot" mask.
    -   **`cam_ids`**: Identifier(s) for the camera(s) from which to record.
    -   **`save_dir`**: The directory where the raw sampled data (videos) for a specific trajectory run will be stored.
    -   **`fps`**: Frames per second for the output videos. Should match your simulation's rendering rate.
    -   **`max_vis_dis`**: Maximum distance (in meters) for depth camera normalization. Depth values beyond this will be clamped. This affects how depth values are scaled in the saved depth video. Default is 1.0 meter.

2.  **Online Sampling (During Simulation Loop)**:
    Inside your simulation loop, after each step or at the desired frame rate, call the `sampling` method.
    ```python
    # Assuming 'sim_node' is an object or interface that provides
    # access to the simulator's current state, including rendering images,
    # depth maps, and segmentation masks.
    # This needs to be adapted to your specific simulator API.
    # For example, sim_node might have methods like:
    # sim_node.get_rgb_image(cam_id)
    # sim_node.get_depth_image(cam_id, max_vis_dis)
    # sim_node.get_segmentation_mask(cam_id) -> returns masks कब्जेक्ट, robot, background

    samples.sampling(sim_node) # Call this repeatedly
    ```
    The `SampleforDR.sampling(sim_node)` method is expected to:
    -   For each camera in `cam_ids`:
        -   Fetch the RGB image.
        -   Fetch the depth image (normalized using `max_vis_dis`).
        -   Fetch segmentation masks that differentiate:
            -   Each object in `objs`.
            -   All parts in `robot_parts` (combined into one robot mask).
            -   The background (everything not an `obj` or `robot_part`).
    -   Store these frames internally.

3.  **Save Collected Data**:
    After the simulation trajectory is complete:
    ```python
    samples.save()
    ```
    This will write out the collected frames as a set of `.mp4` video files in the specified `save_dir`. The expected output files are:
    -   `rgb_<cam_id>.mp4`: The RGB video from the specified camera.
    -   `depth_<cam_id>.mp4`: The normalized depth video.
    -   `mask_<obj_name>_<cam_id>.mp4`: A binary mask video for each object in `objs`.
    -   `mask_robot_<cam_id>.mp4`: A binary mask video for the combined robot parts.
    -   `mask_background_<cam_id>.mp4`: A binary mask video for the background.
    *(The original doc mentions `cam.mp4`, `depth.mp4`, `obj1.mp4`, etc. The naming convention might vary slightly or be configurable, but the content is key.)*

### 5.2 Stage 2: Prompt Generation

Effective domain randomization with generative models requires good text prompts. The `augment.py` script (presumably located in `discoverse/randomain`) helps create these prompts.

-   **Purpose**: To generate a diverse set of textual descriptions that will guide the ComfyUI image generation process. These prompts describe the desired scene, objects, and overall style.

-   **Modes of Operation**:

    1.  **`mode = 'input'` (Batch Generation from Pre-defined Descriptions)**:
        You provide basic descriptions for foreground objects, the robot, the background, a general scene description, and a negative prompt.
        ```python
        # Example for a 'block_place' task in augment.py
        mode = 'input'
        fore_objs = { # Dictionary: {object_name_in_mask: "text description"}
            "block_green": "A green block",
            "bowl_pink": "A pink bowl",
            # robot mask is often handled separately or as part of fore_objs
            "robot": "A black robot arm" 
        }
        background = 'A table' # General background description
        scene = 'In one room, a robotic arm is doing the task of clipping a block into a bowl' # Overall action/context
        negative = "No extra objects in the scene, blurry, low quality" # Things to avoid
        num_prompts = 50 # Number of diverse prompts to generate from these inputs
        ```
        -   The script would then likely use techniques like synonym replacement, template filling, or even LLM-based paraphrasing (if integrated) to generate `num_prompts` variations. For example, "A green block" might become "A vibrant lime-colored cube" or "A small, verdant brick." The background "A table" could become "a wooden desk," "a metal workshop bench," etc. The `scene` prompt provides context.

    2.  **`mode = 'example'` (Augmentation from Example Prompts)**:
        You provide a path to a file (e.g., `example.jsonl`) containing seed prompts.
        ```python
        # Example in augment.py
        mode = 'example'
        input_path = 'path/to/your/example_prompts.jsonl' # File with one JSON object per line, each an example prompt
        num_prompts = 50
        ```
        -   The `example.jsonl` would contain structured prompts, perhaps with placeholders or specific styles. The script then augments these examples to create `num_prompts` variations.
        -   This mode is useful if you have a set of high-quality prompts and want to expand upon them systematically.

-   **Output**: The `augment.py` script will typically output a file (e.g., a `.txt` or `.jsonl` file) containing the list of generated prompts, which will be used by `generate.py`.

### 5.3 Stage 3: Randomized Scene Generation

This is the core step where the sampled data and generated prompts are used to create the final randomized video sequence. This is handled by `discoverse/randomain/generate.py`.

```bash
cd discoverse/randomain
python generate.py [--arg_name arg_value ...]
```

**Key Operational Steps of `generate.py` (Conceptual)**:

1.  **Load Data**: Reads the sampled videos (RGB, depth, masks) for a specific trajectory (`work_dir`) and camera (`cam_id`).
2.  **Load Prompts**: Loads the text prompts generated by `augment.py`.
3.  **Iterate Through Frames/Keyframes**:
    -   For each **keyframe** in the input video (determined by `flow_interval`):
        -   Select a prompt (e.g., round-robin or randomly from the loaded prompts).
        -   Take the corresponding depth frame.
        -   Potentially use object masks to apply different prompts to different regions or to inpaint/outpaint parts of the scene (advanced). The basic usage implies a global prompt conditioned by the overall depth.
        -   Send the depth map and the prompt to ComfyUI (via its API or command-line interface if ComfyUI is run as a server).
        -   ComfyUI uses SDXL Turbo + ControlNet (Depth) to generate a new RGB image that matches the prompt's description while adhering to the depth frame's geometry.
        -   Store the generated randomized frame.
    -   If `flow_interval > 1`:
        -   For frames *between* two generated keyframes:
            -   Use the selected `flow_method` (`rgb` for Farneback, `raft` for RAFT) to compute optical flow between the two keyframes (or the original RGB frames corresponding to the keyframes).
            -   Warp the previously generated keyframe (or a combination of both keyframes) using the computed flow to synthesize the intermediate frame(s).
            -   Store the interpolated frame.
4.  **Save Output**: Combine all generated and interpolated frames into a new randomized output video file.

## 6. Key Parameters for `generate.py`

The following are crucial parameters for `generate.py`. Refer to the script itself for a complete list and default values.

-   **`--task_name`** (e.g., `block_place`):
    -   **Description**: A name for the task, often used for organizing output files or selecting task-specific prompts.
-   **`--work_dir`** (e.g., `000`, `001`):
    -   **Description**: Specifies the subdirectory within `data/randomain/` (or a similar base path) that contains the sampled data for a single trajectory. This usually corresponds to the `save_dir` name used during `SampleforDR`.
-   **`--cam_id`** (e.g., `front_camera`):
    -   **Description**: The identifier of the camera whose sampled data should be processed. This should match one of the `cam_ids` used during sampling.
-   **`--fore_objs`** (e.g., `['block_green', 'bowl_pink', 'robot']`):
    -   **Description**: A list of foreground object names (including the robot if it's to be considered a foreground element for prompting). This helps in potentially associating specific parts of prompts or masks if the generation logic is that granular. It should align with object names used in `SampleforDR` and `augment.py`.
-   **`--wide`, `--height`** (e.g., `1280`, `768`):
    -   **Description**: The width and height for both input processing (if resizing is done) and the output generated images/video. It's recommended to use dimensions supported well by SDXL.
-   **`--num_steps`** (e.g., `4`):
    -   **Description**: Number of diffusion steps for SDXL Turbo. Turbo models are designed for very few steps (e.g., 1-8).
-   **`--flow_interval`** (e.g., `1`, `5`, `10`):
    -   **Description**: Determines how often the full generative model is run.
        -   `1`: Generate every frame using ComfyUI (highest quality/consistency, slowest).
        -   `N > 1`: Generate one frame using ComfyUI every `N` frames. The `N-1` intermediate frames are generated using the specified `flow_method`.
-   **`--flow_method`** (e.g., `'rgb'`, `'raft'`):
    -   **Description**: Specifies the optical flow algorithm to use when `flow_interval > 1`.
        -   `'rgb'`: Farneback method.
        -   `'raft'`: RAFT method (requires RAFT model to be deployed).

## 7. Output

The primary output of the `generate.py` script will be:
-   **Randomized Video(s)**: Located typically within a subdirectory of the `work_dir` (e.g., `data/randomain/000/randomized_front_camera.mp4`). These videos contain the visually altered scene.
-   **Possibly intermediate files**: Depending on the implementation, it might also save individual generated frames or flow fields.

The goal is to use these randomized videos as augmented training data for downstream tasks, such as training robot policies or perception systems.



