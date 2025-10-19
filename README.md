# PrismXL Image Generator

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PySide6-brightgreen.svg)](https://www.qt.io/qt-for-python)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.21+-yellow.svg)](https://github.com/huggingface/diffusers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="https://github.com/user-attachments/assets/6e1fc9c4-f134-40f1-8345-6da8e155cba0" alt="Application Demo Video" />
    </td>
    <td width="50%" valign="top">
      <p>A high-performance, feature-rich desktop GUI for generating images using state-of-the-art diffusion models. Built with PySide6 for a responsive and native user experience, this application provides a robust interface for both casual users and advanced artists to create stunning visuals.</p>
    </td>
  </tr>
</table>



| | |
|:---:|:---:|
| <img width="635" alt="Screenshot 2025-10-19 135004" src="https://github.com/user-attachments/assets/dd1301c8-6a71-418c-851a-a252bf8831b8" /> | <img width="635" alt="Screenshot 2025-10-19 133740" src="https://github.com/user-attachments/assets/7370da5f-a395-4f95-814e-b3c1c9f79b9f" /> |
| <img width="635" alt="Screenshot 2025-10-19 133318" src="https://github.com/user-attachments/assets/7c44f8e1-d17d-4409-9844-9f2a9c55d15b" /> | <img width="635" alt="Screenshot 2025-10-19 132924" src="https://github.com/user-attachments/assets/dbb8b7d9-aeb5-44c7-9f0a-5ef854e81c3d" /> |


## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
- [Configuration and Data](#configuration-and-data)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Disclaimer](#disclaimer)

## Introduction

The PrismXL Image Generator is a standalone desktop application designed to harness the power of diffusion models, specifically leveraging the `RunDiffusion/Juggernaut-XL-v9` model. It offers a comprehensive suite of tools for crafting the perfect image, from basic text-to-image generation to fine-grained control over advanced parameters. The application is built with a focus on performance, responsiveness, and user experience, featuring asynchronous generation to prevent UI freezes, real-time system resource monitoring, and a host of workflow-enhancing utilities.

## Key Features

### Generation Engine
- **High-Quality Model**: Natively integrates the powerful `Juggernaut-XL-v9` diffusion model.
- **Advanced Parameter Control**: Adjust Inference Steps, CFG Scale, and CLIP Skip for precise creative control.
- **Batch & Grid Generation**: Generate multiple images from a single prompt to explore variations efficiently.
- **Multiple Resolutions**: Supports a wide range of resolutions, including standard and widescreen formats.
- **Reproducibility**: Use custom seeds to recreate previous results.
- **Optimized Performance**: Automatically enables VAE tiling for high-resolution images to conserve memory.
- **Hardware Acceleration**: Utilizes CUDA for GPU-accelerated generation with a fallback to CPU if needed.

### User Interface
- **Modern & Responsive UI**: Built with PySide6 for a clean, fast, and native cross-platform experience.
- **Custom Frameless Window**: A sleek, modern design with a custom title bar.
- **Light & Night Modes**: Switch between themes for your visual comfort.
- **Image Viewer**: Displays generated images in a grid with thumbnails and a detailed main view.
- **Magnifying Loupe**: An interactive zoom tool to inspect image details directly in the viewer.
- **Modular Layout**: Re-arrange UI sections via drag-and-drop to customize your workspace.

### Workflow & Tools
- **Prompt Library**: Save, categorize, search, and reuse your favorite prompts.
- **Built-in Spell Checker**: Catches typos in your prompts and offers corrections to improve generation quality.
- **Real-time Progress**: Monitor generation progress with a detailed progress bar, status updates, and time estimates.
- **System Resource Monitor**: Keep an eye on RAM and GPU memory usage directly within the application.
- **Live Rendering**: (Optional) Watch your image come to life with a real-time preview of the generation process for 512x512 images.
- **Metadata Saving**: Automatically saves generation parameters (prompt, seed, steps, etc.) in a JSON file alongside the saved image.
- **Persistent Settings**: The application remembers your UI layout, theme, and generation settings between sessions.

## Technical Stack

- **Backend**: Python
- **GUI Framework**: PySide6 (The official Qt for Python project)
- **AI/ML**:
  - PyTorch
  - Hugging Face Diffusers
  - Transformers
  - Accelerate
- **Image Processing**: Pillow (PIL)
- **System Utilities**: Psutil, GPUtil
- **Miscellaneous**: Pyspellchecker

## Installation

### Prerequisites
1.  **Python**: Version 3.9 or newer.
2.  **Git**: For cloning the repository.
3.  **NVIDIA GPU (Recommended)**: For hardware-accelerated generation. Ensure you have the latest NVIDIA drivers and a compatible CUDA Toolkit version installed.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PrismXL-Image-Generator.git
    cd PrismXL-Image-Generator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    A `requirements.txt` file should be provided. To ensure PyTorch is installed with the correct CUDA support for your system, it is highly recommended to first visit the [PyTorch website](https://pytorch.org/get-started/locally/) and install it using the command provided there.
    
    Example for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```    
    Then, install the remaining dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    *If a `requirements.txt` is not available, install the core packages:*
    ```bash
    pip install PySide6 diffusers transformers accelerate torch numpy Pillow psutil GPUtil pyspellchecker safetensors
    ```

## Usage

Once the installation is complete, run the application from the root directory of the project:

```bash
python main.py
```

The main window will appear. Follow these steps to generate your first image:
1.  **Enter a Prompt**: Type a description of the image you want to create in the "Prompt" text area.
2.  **Enter a Negative Prompt (Optional)**: Type concepts you wish to exclude in the "Negative Prompt" area.
3.  **Adjust Settings (Optional)**: Use the sliders and dropdowns in the "Advanced Options" section to fine-tune the generation process.
4.  **Generate**: Click the "Generate" button. The UI will remain responsive while the image is being created.
5.  **View and Save**: The generated image(s) will appear on the right. You can select a thumbnail to view it larger, and use the "Save" button to save the selected image to your computer. Right-click the "Save" button to save all images from the current grid.

## Configuration and Data

- **Settings**: The application saves UI and user preferences automatically. On Windows, these settings are stored in the registry under `HKEY_CURRENT_USER\Software\Sapphire\PrismXL`.
- **Image Autosaves**: All generated images are automatically saved for your convenience in a user-specific directory:
  - **Windows**: `C:\Users\<YourUsername>\.sapphire_prismxl\images\`
  - **macOS/Linux**: `/home/<YourUsername>/.sapphire_prismxl/images/`
- **Prompt Library**: Your saved prompts are stored in `prompt_library.json` within the same `.sapphire_prismxl` directory.
- **Logs**: A detailed log file, `image_generator.log`, is created in the application's root directory for troubleshooting purposes.

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code follows existing style conventions and includes comments where necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- This project is built upon the incredible work of the teams behind [PyTorch](https://pytorch.org/), [Hugging Face](https://huggingface.co/), and the [Qt for Python (PySide6)](https://www.qt.io/qt-for-python) project.
- The core image generation capabilities are powered by the `diffusers` library.
- The AI model `Juggernaut-XL-v9` was created by [RunDiffusion](https://rundiffusion.com/).

## Disclaimer

This software provides an interface to a third-party generative AI model. The developer of this application is not the creator of the underlying image generation model.

- The developer of this tool takes **zero responsibility or liability** for any content or images generated by the user.
- All outputs are the sole responsibility of the user.
- It is the user's responsibility to adhere to all applicable local, national, and international laws, regulations, and ethical guidelines regarding the use of generative AI and the content they create.
