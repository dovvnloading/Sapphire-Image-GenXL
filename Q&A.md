# PrismXL - Frequently Asked Questions (FAQ)

Welcome to the PrismXL FAQ! If you can't find an answer to your question here, please consult the full [User Guide](https://github.com/dovvnloading/Sapphire-Image-GenXL/blob/main/User_Guide.md).

## Table of Contents
1.  [General Questions](#general-questions)
2.  [Usage & Features](#usage--features)
3.  [Improving Your Images](#improving-your-images)
4.  [Troubleshooting & Technical Issues](#troubleshooting--technical-issues)
5.  [Project & Development](#project--development)

---

## General Questions

#### Q: What is PrismXL?
**A:** PrismXL is a free, open-source desktop application that provides a user-friendly interface for creating AI-generated images. It uses a powerful AI model to turn your text descriptions (prompts) into unique pictures.

#### Q: What AI model does PrismXL use?
**A:** The current version of PrismXL uses the **Juggernaut-XL-v9** model, which is a fine-tuned version of a Stable Diffusion XL (SDXL) model. It is known for its high-quality, detailed, and artistic outputs.

#### Q: Is PrismXL completely free to use?
**A:** Yes. PrismXL is free and open-source software. You can use it to generate as many images as you like, and you can view the source code to see how it works. Please refer to the `LICENSE` file for details on what you can do with the software.

#### Q: Do I need an internet connection to use it?
**A:** The initial setup and model download will require an internet connection. After the model is downloaded and cached by the `diffusers` library, you can run the application offline.

---

## Usage & Features

#### Q: Where are my images saved?
**A:** PrismXL uses two saving methods:
1.  **Automatic Save:** Every image you generate is automatically saved in a user-specific folder to prevent accidental loss. You can find this folder at: `C:\Users\[YourUsername]\.sapphire_prismxl\images\`
2.  **Manual Save:** When you click the **Save** button, you can choose exactly where on your computer to save the currently selected image.

#### Q: How do I save all the images from a grid at once?
**A:** **Right-click** on the **Save** button. This will open a dialog box asking you to select a directory, and all images currently in the thumbnail strip will be saved there.

#### Q: What is a "Seed" and why should I care about it?
**A:** The seed is the starting number for the random generation process. Think of it as the ID for a specific image generation.
*   **If you use the same prompt, settings, and seed**, you will get the exact same image every time.
*   This is incredibly useful if you create an image you love and want to experiment with small changes to the prompt without losing the original composition.

#### Q: How can I reuse a prompt I really liked?
**A:** Use the **Prompt Library**. Right-click in the main prompt text box and select "Save to Library." Give it a title. Later, you can open the library from the top menu (`Library > Open Prompt Library`), find your prompt, and click "Cast" to send it back to the main window.

#### Q: Why can't I use the "Live Render" feature?
**A:** The live render feature, which shows you a preview as the image generates, is computationally intensive and is currently optimized to work only for **512x512 resolution** images. If you select a different resolution, the option will have no effect.

#### Q: Can I rearrange the sections in the left panel?
**A:** Yes! The "Prompt," "Advanced Options," and other sections are movable. Simply click and hold the title of a section (like "Advanced Options") and drag it up or down to reorder the panel to your liking.

---

## Improving Your Images

#### Q: My images look blurry, generic, or low quality. How can I improve them?
**A:** Image quality depends heavily on your prompt and settings. Here are some tips:
1.  **Be More Descriptive:** Instead of "a dog," try "photo of a golden retriever puppy playing in a field of flowers, cinematic lighting, sharp focus."
2.  **Use "Magic Words":** Add keywords like `masterpiece`, `4k`, `8k`, `highly detailed`, `photorealistic`, `trending on artstation` to your prompt.
3.  **Increase Steps:** A higher number of steps (try 30-50) gives the AI more time to refine the image details.
4.  **Adjust CFG Scale:** A value between 5 and 9 usually provides a good balance between following your prompt and maintaining creative quality.

#### Q: The AI generated weird hands, faces, or extra limbs. How do I stop this?
**A:** This is a common challenge in AI image generation. The best tool to fix this is the **Negative Prompt**. Add descriptive terms for what you want to avoid.
*   **Good Negative Prompt Examples:** `disfigured, malformed hands, extra limbs, extra fingers, ugly, poorly drawn face, blurry, watermark, text, signature`

#### Q: What do the "CFG Scale" and "Steps" sliders actually do in simple terms?
**A:**
*   **Steps Slider:** Think of this as "thinking time." More steps allow the AI to spend more time refining the image from noise into a detailed picture. Too few steps can look unfinished; too many may not add much more detail for the extra time.
*   **CFG Scale Slider:** Think of this as a "creativity vs. obedience" knob. A low value gives the AI more creative freedom, which can sometimes ignore parts of your prompt. A high value forces the AI to follow your prompt very strictly, which can sometimes lead to less artistic or strange results.

---

## Troubleshooting & Technical Issues

#### Q: I got a "CUDA out of memory" error. What does this mean and how do I fix it?
**A:** This is the most common error and it means your graphics card (GPU) ran out of video memory (VRAM). Here are the solutions, from easiest to most effective:
1.  **Close other programs:** Web browsers, games, or other GPU-intensive applications can consume VRAM.
2.  **Lower the Resolution:** This is the biggest factor. Generating a 1024x1024 image requires significantly more memory than a 512x512 one.
3.  **Reduce the Number of Images:** Change the "Number of Images" slider to 1. Generating a grid of 4 images at once uses much more memory.
4.  **Restart the Application:** This can clear any fragmented memory in your VRAM.

#### Q: The application is running very slowly. Is this normal?
**A:** Yes, AI image generation is a very demanding task. The speed depends almost entirely on the power of your graphics card (GPU). Generation on a CPU is possible but will be extremely slow (many minutes per image). The biggest factors affecting speed are image resolution and the number of steps.

#### Q: What are the minimum system requirements?
**A:**
*   **OS:** Windows 10/11, macOS, or Linux.
*   **RAM:** 16 GB of system RAM is recommended.
*   **GPU:** For a good experience, a dedicated NVIDIA GPU with at least **6 GB of VRAM** is strongly recommended (8 GB+ is ideal). The application may run on less powerful GPUs or CPUs, but performance will be very slow.
*   **Disk Space:** At least 10 GB of free space for the AI models and libraries.

#### Q: The application crashed. What should I do?
**A:** First, find the log file named `image_generator.log` in the application's main directory. This file contains valuable error information. Then, please report the issue on the project's GitHub "Issues" page, and if possible, copy and paste the relevant error messages from the log file.

---

## Project & Development

#### Q: Can I change the AI model to a different one (e.g., from Civitai)?
**A:** Currently, PrismXL is hardcoded to use the Juggernaut-XL-v9 model for simplicity and reliability. The ability to load custom community models is a planned feature for a future release.

#### Q: I have an idea for a feature or found a bug. How can I contribute?
**A:** That's fantastic! Please check out our `CONTRIBUTING.md` file, which has detailed instructions on how to report bugs, suggest features, and contribute code to the project. We welcome community involvement
