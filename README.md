📌 Project Description

This project is a Python-based image processing application developed as part of a midterm exam. The application automatically detects image files placed in an input directory, applies multiple image processing techniques, and saves the processed images into an output directory.

To demonstrate DevOps automation and collaboration, the project is integrated with a GitHub Actions Continuous Integration (CI) pipeline that runs automatically on every push to the repository.


🛠️ Tools and Technologies Used
    • Python 3
    • OpenCV (opencv-python)
    • GitHub & GitHub Actions
    • PyTest


🧪 Image Processing Techniques Used

The following image processing features are implemented:

1. Image File Size Reducer
    - Compresses images to reduce file size while maintaining acceptable quality.

2. Watermark Adder
    - Adds a watermark to the image.
    - Note: The watermark depends on the resolution of the input image to ensure proper scaling.

3. Fish Eye Filter
    - Applies a distortion effect that simulates a fisheye lens.

4. 3D Anaglyph Filter
    - Creates a red-cyan anaglyph effect to simulate a 3D appearance.

5. Geometry Filter
    - Converts images into geometrical shape representations (such as polygons, blocks, or abstract geometric forms).


📂 Supported Image File Types

The application only processes the following image formats:
  • jpg
  • jpeg
  • png
Any other file types will be ignored automatically.


⚙️ GitHub Actions CI Pipeline

This project uses **GitHub Actions** to implement a Continuous Integration (CI) pipeline.

The CI pipeline is defined in a YAML (`.yml`) file and automatically runs on every push to the GitHub repository.

CI Pipeline Workflow
The GitHub Action performs the following steps:
  1. Checks out the repository code
  2. Sets up the Python environment
  3. Installs required dependencies
  4. Runs automated tests using PyTest
  5. Verifies that the image processing scripts execute successfully

