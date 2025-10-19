# PrismXL User Guide

Welcome to PrismXL! This guide will walk you through everything you need to know to start creating stunning AI-generated images.

## Table of Contents
1.  [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [First Launch: Terms of Use](#first-launch-terms-of-use)
2.  [Understanding the Interface](#understanding-the-interface)
    *   [The Control Panel (Left)](#the-control-panel-left)
    *   [The Image Viewer (Right)](#the-image-viewer-right)
3.  [Creating Your First Image: A Step-by-Step Guide](#creating-your-first-image-a-step-by-step-guide)
4.  [Detailed Feature Breakdown](#detailed-feature-breakdown)
    *   [Prompt Section](#prompt-section)
    *   [Advanced Options](#advanced-options)
    *   [Progress & System Resources](#progress--system-resources)
    *   [The Top Menu Bar](#the-top-menu-bar)
5.  [Managing Your Prompts: The Prompt Library](#managing-your-prompts-the-prompt-library)
6.  [Tips for Best Results](#tips-for-best-results)
7.  [Troubleshooting](#troubleshooting)

---

## 1. Getting Started

### Installation
PrismXL is a standalone application. To install it, simply run the provided installer or place the executable file in a directory of your choice. No complex setup is required.

### First Launch: Terms of Use
The first time you launch PrismXL, you will be presented with a **Terms of Use & Liability Disclaimer**. You must read and agree to these terms to proceed. This is a crucial step to ensure you understand your responsibilities as a user.

---

## 2. Understanding the Interface

The main window of PrismXL is split vertically into two primary sections. On the left is the **Control Panel**, where you input your commands and adjust settings. On the right is the **Image Viewer**, where your creations appear.

### The Control Panel (Left)
This is your mission control for image generation. It contains several titled sections that you can collapse, expand, or even rearrange by dragging and dropping them to customize your workspace.

*   **Prompt**: The section at the top where you describe the image you want to create.
*   **Advanced Options**: A collection of sliders and menus to fine-tune the generation process.
*   **Progress Information**: Tools to monitor the status of your image generation in real-time.
*   **System Resources**: A display to keep an eye on your computer's performance.

### The Image Viewer (Right)
This large area is dedicated to displaying your generated images.

*   **Main Display**: Shows the currently selected image in a large format.
*   **Thumbnail Strip**: When you generate multiple images at once, a strip of small preview images (thumbnails) will appear at the bottom. Clicking a thumbnail will display it in the Main Display.

---

## 3. Creating Your First Image: A Step-by-Step Guide

Let's jump right in and create an image.

1.  **Write a Prompt**: In the large **Prompt** text box at the top of the Control Panel, type a description of what you want to see. Be descriptive!
    *   *Example: "A majestic lion with a crown made of stars, sitting on a throne on the moon, hyperrealistic."*

2.  **(Optional) Add a Negative Prompt**: In the smaller **Negative Prompt** box below the main one, type what you want to *avoid*.
    *   *Example: "blurry, cartoon, disfigured, text, watermark"*

3.  **Adjust Settings (Optional)**: For your first image, the default settings in the **Advanced Options** section are a great starting point.

4.  **Generate**: Click the **Generate** button, located just below the prompt boxes.

5.  **Watch the Magic**: In the **Progress Information** section, the progress bar will fill up, and the status label will show you what's happening. Once complete, your image will appear in the Image Viewer on the right.

6.  **Save Your Creation**: If you love the result, click the **Save** button to save the image to your computer.

---

## 4. Detailed Feature Breakdown

### Prompt Section
This section is located at the top of the Control Panel.

*   **Prompt Box**: The main text area for your image description. It features a built-in spell checker. Misspelled words will be underlined with a red wave. Right-click on a word for suggestions.
*   **Negative Prompt Box**: Use this to exclude elements or styles from your image. It also has a spell checker.
*   **Generate Button**: Starts the image generation process.
*   **Cancel Button**: Becomes active during generation. Click it to stop the current process.
*   **Save Button**: Saves the currently selected image. **Right-click** this button to save all images from a generated grid at once.

### Advanced Options
This section contains several sliders and drop-down menus to give you precise control.

*   **Steps**: A slider that controls the number of refinement steps the AI takes. Drag it to the right for more steps, which can increase detail but also takes longer. A recommended range is 25-50.

*   **Number of Images**: A slider to set how many images (up to 4) are generated in a single batch. This is great for exploring variations of a prompt.

*   **CFG Scale (Classifier Free Guidance)**: A slider that determines how strictly the AI follows your prompt. Lower values (3-6) allow for more creativity. Higher values (7-12) stick closer to your prompt but can feel rigid.

*   **CLIP Skip**: A slider that can alter the style of an image by skipping final layers of the text analysis model. A setting of 1 or 2 is common for subtle changes.

*   **Resolution**: A drop-down menu to set the width and height of your image in pixels. Higher resolutions require more computer memory and take longer to generate.

*   **Custom Seed**: A checkbox and a text field. A seed is the starting number for the random generation process.
    *   **Unchecked**: A random seed is used every time, producing different results.
    *   **Checked**: You can enter a specific number in the text field. Using the same prompt and seed will produce the exact same image again, which is useful for recreating results.

### Progress & System Resources
These sections provide real-time feedback during generation.

*   **Progress Bar**: A visual bar that fills from left to right to track the generation progress.
*   **Status Label**: A text display that gives real-time updates (e.g., "Loading model", "Step 5/30").
*   **Timer Labels**: Shows the elapsed time for the current generation and an estimated time remaining.
*   **RAM/GPU Usage**: Displays your system's memory usage, helping you manage performance.

### The Top Menu Bar
At the very top of the application window, you will find several menu options.

*   **Settings**:
    *   **Night Mode**: Toggles between a light and dark theme for the interface.
    *   **Live Render**: (For 512x512 images only) When checked, shows a blurry preview of the image as it's being generated.
    *   **Close**: Exits the application.

*   **Window**: Allows you to show or hide the different sections of the Control Panel (Prompt, Advanced Options, etc.) to customize your view.

*   **Library**:
    *   **Open Prompt Library**: Opens a new window to manage your saved prompts.

---

## 5. Managing Your Prompts: The Prompt Library

Tired of retyping your favorite prompts? Use the Prompt Library!

*   **Saving a Prompt**:
    1.  Type a prompt in the main **Prompt** box.
    2.  Right-click inside the box and select "Save to Library" from the context menu.
    3.  A dialog box will appear asking you to give your prompt a memorable title.

*   **Using a Saved Prompt**:
    1.  Click **Library > Open Prompt Library** from the top menu bar. A new window will appear.
    2.  Find the prompt you want to use. You can scroll through the list or use the search bar at the top.
    3.  Select it and click the **Cast** button at the bottom. The prompt will be sent to the main window.

*   **Editing/Deleting**: In the library window, you can select a prompt and click the **Edit** or **Delete** buttons to manage your collection.

---

## 6. Tips for Best Results

*   **Be Specific**: Instead of "a car," try "a vintage red sports car from the 1960s, shiny, parked on a cobblestone street at sunset."
*   **Use Adjectives**: Words like `masterpiece`, `hyperdetailed`, `cinematic lighting`, `4k`, `sharp focus` can significantly improve image quality.
*   **Combine Concepts**: Don't be afraid to mix and match creative ideas.
*   **Iterate**: Generate a grid of 4 images using the "Number of Images" slider. Find one you like, then check the "Custom Seed" box and use its seed with slight prompt modifications to refine it further.

---

## 7. Troubleshooting

*   **Error Message about "CUDA" or "Memory"**: This usually means your GPU ran out of memory.
    *   **Solution**: Close other memory-intensive applications, lower the image resolution from the drop-down menu, or reduce the "Number of Images" slider to 1.

*   **Application is Slow**: High resolutions and a high number of steps are computationally expensive. This is normal. If it's too slow for you, reduce these settings.

*   **Generated Images Look Strange**: This can be part of the creative process! Try modifying your prompt, using a negative prompt to exclude unwanted features (e.g., "extra limbs, ugly"), or changing the CFG Scale.

***

Thank you for using PrismXL. Happy creating
