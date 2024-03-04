![UltraDarkFusion GUI](fusion.gif)

**UltraDarkFusion** is an advanced GUI application tailored for object detection, image processing, and computer vision. It simplifies workflows for collection, labeling, dataset management, model training, and visualization of real-time inferences.

## Features
- <b>Annotation Tools:</b> Customizable bounding box annotation with YOLO format conversion.
- <b>Dataset Management:</b> Tools for duplicate removal, image splitting, and dataset preparation.
- <b>Image Augmentation:</b> Apply effects such as motion blur, noise, and lighting adjustments.
- <b>Framework Integration:</b> Compatible with Darknet and Ultralytics frameworks.
- <b>Model Training:</b> In-built training capabilities for Ultralytics and Darknet models.
- <b>Visualization:</b> Automatic labeling and visualization using pre-trained .weights or .pt files for efficient model evaluation.
- <b>Performance Metrics:</b> Analytics to monitor and optimize training progress.
- <b>Hardware Acceleration:</b> Support for CUDA, OpenCL, and CPU backends.
- <b>Customizable UI:</b> Themes, styling, and user-defined shortcuts.
- <b>Video Processing:</b> Download YouTube videos, extract frames, or collect images from the screen directly.
- <b>Segment Anything:</b> Correct bounding boxes to fit around your objects.
- <b>Grounding-DINO:</b> Label anything in your classes.txt.
- <b>TensorRT Support:</b> Support for .engine on Ultralytics YOLOv8.



UltraDarkFusion is designed to be modular, enabling easy integration of new algorithms and streamlining the process from dataset preparation to model training.

## 🚀 Getting Started with Installation

Before diving into the code, let's set up your environment with the necessary build tools. Open up a command prompt and execute the following commands to download Git and CMake. These are essential tools for version control and building the project, respectively Install Git and cmake close the terminal after your done this allows it to refresh otherwise you may recieve and error.
## Installation Guide

There are two options for installing UltraDarkFusion - Option #1 "OpenCL without CUDA" is the simplier of the two but you may see a slight reduction in performance, Option #2 "DNN with CUDA" is more complex to setup but will likely perform slightly better.  Both options require you to install darknet and it requires you to install CUDA and cuDNN.  Be advised that if you want the installation to go smoothly, you MUST use the exact versions as shown below and the exact folder structure as shown below. You may customize your installation but you will need to make complex modifications to it.

### Option 1: OpenCL without CUDA

This guide outlines the installation of UltraDarkFusion without opencv_cuda.

#### Preperation

   ```sh
   winget install git.git
   winget install Kitware.CMake
   winget install Microsoft.VisualStudio.2022.Community
  ```
 

 **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.
    
      click on the "Windows Start" menu and run "Visual Studio Installer"
      click on Modify
      select Desktop Development With C++
      select python development
      click on Modify in the bottom-right corner, and then click on Yes
      After installation, A system restart is required.  ```


#### Installation

### 1. Clone UltraDarkFusion Repository:
   Clone the repository:
   ```sh
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   mkdir anaconda
   ```
2. **Install Weights From Shared Google Drive**: 
   - Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1JLH7kMc6FXdKz1fmoxO5AfLhASDzmpw8/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

   **OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing

2. (Alternative) **Install Weights From Source**: 
   - In the `ultradarkfusion` folder, create a folder called `Sam` and download the checkpoints from [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
   - For Grounding DINO checkpoints: [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO).

3. **Download Anaconda**:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.
          
4. **Run the Anaconda Installer**:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:\DarkFusion\anaconda`.

5. **Execute fusion_install.bat**:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your DarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

6. **Install Darknet**:
   - There are various verions of Darknet available, at the time this readme was written, the HANK-AI/Darknet version was the most active and is the suggested version.  You can elect to use a different version, but your setup might be more complex
      # Darknet Options

      - [Darknet by Hank-AI](https://github.com/hank-ai/darknet) (Currently maintained as of 10/22/2023)
      Discord: [https://discord.gg/fZTz8E44](https://discord.gg/fZTz8E44)

      - [Darknet by Umbralada](https://github.com/umbralada/darknet) (Recently updated as of 10/22/2023)

      - [Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) (No longer maintained)

      > **Disclaimer**: UltraDarkFusion may not fully support all versions of Darknet for visual UI output, although all versions are suitable for training. This is due to variances in output formats across different Darknet versions.

   - In Hank-AI/Darknet There are various steps in the "Windows CMake Method that are redundant with steps in this installation 
    - you will not need to instal Git again
    - You will not need to install CMAKE again
    - You will not need to install Microsoft Visual Studio 2022 Community again
    - You will not need to modify your Visual Studio to include "Desktop Development with C++" again
    - You WILL need to install NSIS
    - Use CUDA Version 11.8.0_522.06_windows - Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Use cuDNN version 8.7.0.84_cuda10 - [NVIDIA Developer](https://developer.nvidia.com/cudnn).
      - Note: cuDNN versions prior to 9 do not have an excutable installer, you must manually move the files into the right directories and manually add the environment variables path
         - The folder you extract to should look exactly like this: C:\Program Files\NVIDIA\CUDNN\v8.x
         - Your system environment path should look like this: 
            - Variable Name: Path
            - C:\Program Files\NVIDIA\CUDNN\v8.x\bin

This completes the "Option 1: OpenCL without CUDA" installation.

*Recommended:* Use Visual Studio Code for a smoother experience with the program.Download it here https://code.visualstudio.com/Download

If you prefer not to use Visual Studio Code:

Open Anaconda Prompt.
Activate the fusion environment by running:
`conda activate fusion`
Navigate to the UltraDarkFusion directory:
`cd C:\DarkFusion\UltraDarkFusion`
Use the Tab key to auto-complete and select UltraDarkFusion_X.py, then press Enter to start the program.
For further instructions, refer to the help tab in the program or join the Discord community.
or simply just double click the .py

### Option 2: DNN with CUDA

#### Preperation

Before proceeding with the installation, it's essential to assess your existing development environment. If you have a functioning setup with OpenCV and CUDA support, along with PyTorch, please consult the requirements.txt file for a list of required packages—install only the necessary ones.

Caution: It's recommended to run the program using OpenCL as there isn't a significant performance boost when utilizing .weights and .cfg files on CUDA, and the setup can be challenging. While PyTorch is installed and all .pt files are utilizing CUDA from PyTorch, compiling with opencv_cuda can be quite intricate.

## Prerequisites: Clean Your PC

To ensure a smooth installation process when setting up OpenCV with CUDA support, follow these actions:

1. **Uninstall Python**: Remove any existing Python installations from the Control Panel.

2. **Delete Python Cache**: Navigate to `%USERPROFILE%\AppData\Local\Programs\python` and remove python.

4. **Remove CUDA 11.x**: Uninstall all versions of CUDA 11.x from the Control Panel. If the folder `C:\Program Files\NVIDIA GPU Computing Toolkit` remains, delete it.

5. **Update NVIDIA Drivers**: Either update your GeForce Experience Game Driver or visit [NVIDIA's driver download page](https://www.nvidia.com/Download/index.aspx?lang=en-us).

6. **Ensure you have at least 30 GB of free space for the installation.**


   ```sh
   winget install git.git
   winget install Kitware.CMake
   winget install Microsoft.VisualStudio.2022.Community
  ```
 

 **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.
    
      click on the "Windows Start" menu and run "Visual Studio Installer"
      click on Modify
      select Desktop Development With C++
      select python development
      click on Modify in the bottom-right corner, and then click on Yes
      After installation, A system restart is required.  ```


#### Installation

1. Clone UltraDarkFusion Repository:
   Warning: If a different version of OpenCV and contrib are already installed in your environment, ensure they are updated to match the required version for UltraDarkFusion.

   Begin by navigating to your desired installation directory (e.g., C:/) using the command prompt. Following that, execute the commands below to clone the UltraDarkFusion repository and the specified versions of OpenCV and contrib:
   ```sh
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   git clone --branch 4.7.0 https://github.com/opencv/opencv.git opencv-4.7.0
   git clone --branch 4.7.0 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.7.0
   mkdir anaconda build install 
   ```

2. **Install Weights From Shared Google Drive**: 
   - Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1JLH7kMc6FXdKz1fmoxO5AfLhASDzmpw8/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

   **OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing

2. (Alternative) **Install Weights From Source**: 
   - In the `ultradarkfusion` folder, create a folder called `Sam` and download the checkpoints from [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
   - For Grounding DINO checkpoints: [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO).

3. **Download Anaconda**:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.

4. **Run the Anaconda Installer**:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:\DarkFusion\anaconda`.

5. **Execute fusion_install.bat**:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your UltraDarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

6. **Install Darknet**:
   - There are various verions of Darknet available, at the time this readme was written, the HANK-AI/Darknet version was the most active and is the suggested version.  You can elect to use a different version, but your setup might be more complex
      # Darknet Options

      - [Darknet by Hank-AI](https://github.com/hank-ai/darknet) (Currently maintained as of 10/22/2023)
      Discord: [https://discord.gg/fZTz8E44](https://discord.gg/fZTz8E44)

      - [Darknet by Umbralada](https://github.com/umbralada/darknet) (Recently updated as of 10/22/2023)

      - [Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) (No longer maintained)

      > **Disclaimer**: UltraDarkFusion may not fully support all versions of Darknet for visual UI output, although all versions are suitable for training. This is due to variances in output formats across different Darknet versions.

   - In Hank-AI/Darknet There are various steps in the "Windows CMake Method that are redundant with steps in this installation 
    - you will not need to instal Git again
    - You will not need to install CMAKE again
    - You will not need to install Microsoft Visual Studio 2022 Community again
    - You will not need to modify your Visual Studio to include "Desktop Development with C++" again
    - You WILL need to install NSIS
    - Use CUDA Version 11.8.0_522.06_windows - Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Use cuDNN version 8.7.0.84_cuda10 - [NVIDIA Developer](https://developer.nvidia.com/cudnn).
      - Note: cuDNN versions prior to 9 do not have an excutable installer, you must manually move the files into the right directories and manually add the environment variables path
         - The folder you extract to should look exactly like this: C:\Program Files\NVIDIA\CUDNN\v8.x
         - Your system environment path should look like this: 
            - Variable Name: Path
            - C:\Program Files\NVIDIA\CUDNN\v8.x\bin
    
6. **cuDNN Files**:
   - Copy the files you extracted to here `C:\Program Files\NVIDIA\CUDNN\v8.x` during the installation of Darknet to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.

6. (Alternative) **cuDNN Files**:
   - Download directly from my Google Drive [here](https://drive.google.com/file/d/1PIdG6qZnyfhNFF7vUVNoX5fDMtwgy5uJ/view?usp=sharing).

7. Compile OpenCV with CUDA:
   - Run the `cuda.bat` script to compile OpenCV with CUDA support.

8. **Install Weights From Shared Google Drive**: 
   - Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1JLH7kMc6FXdKz1fmoxO5AfLhASDzmpw8/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

   **OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing

8. (Alternative) **Install Weights From Source**: 
   - In the `ultradarkfusion` folder, create a folder called `Sam` and download the checkpoints from [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
   - For Grounding DINO checkpoints: [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO).

9. **Installing TensorRT from Source**
   - Install TensorRT from source for advanced optimizations. Follow these steps to install TensorRT into the `C:\DarkFusion\UltraDarkFusion` directory:
      - Download the TensorRT zip file from NVIDIA: 
      [TensorRT-8.6.1.6 for Windows 10](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip)
      - Extract the contents of the zip file to `C:\DarkFusion\UltraDarkFusion`.
      - Open a anaconda prompt and change the directory to the TensorRT folder:
         ```sh
         conda activate fusion
         cd C:\DarkFusion\UltraDarkFusion\TensorRT-8.6.1.6
  

   - Install the required TensorRT wheels using pip. Make sure to install them in the following order:
      ```aql
      pip install python\tensorrt-8.6.1-cp38-none-win_amd64.whl
      pip install graphsurgeon\graphsurgeon-0.4.6-py2.py3-none-any.whl
      pip install uff\uff-0.6.9-py2.py3-none-any.whl
      pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
      pip install protobuf==3.20.3
      ```
   - copy the files from `tensorrt-8.6.1.6\lib` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`

# Ultralytics

Ultralytics is pre-installed as part of this package. For documentation, visit the [Ultralytics documentation site](https://docs.ultralytics.com/).

---

**CONTRIBUTIONS AND SUPPORT**

A special shoutout to everyone who has contributed to the development and improvement of UltraDarkFusion. Your insights and support have been invaluable.

**UltraDarkFusion Discord:** Join us for discussions, support, and community updates: [https://discord.gg/kGaWChbUtR](https://discord.gg/kGaWChbUtR)

Many thanks to all who helped and to all who had input on this. 
Long call #1 tester, thanks for all your help.
see Discord for links to their Discord servers and websites.

Main inspiration was from DarkMark, a Linux-based label maker maintained by Stephane, the current developer for the Hank-AI Darknet repo.
**DarkMark GitHub:** [https://github.com/stephanecharette/DarkMark](https://github.com/stephanecharette/DarkMark)

Label Maker Pro (thanks EAL)!
**Label Maker Pro Search:** [https://www.bing.com/search?q=easy+aimlock](https://www.bing.com/search?q=easy+aimlock)

Outside of Darknet and Ultralytics, I utilized SAM. See GitHub:
**SAM - Segment Anything Model:** [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 
**GroundingDINO:** [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for additional tools used in my projects.

**DarkFusion**


