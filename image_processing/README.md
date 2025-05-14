🖼️ Image Processing & Classification Project
This project implements basic and advanced image processing techniques from scratch in Python, including image negation, grayscale transformation, rotation, blurring, edge detection, chroma keying, and sticker overlays.
It also includes a simple K-Nearest Neighbors (KNN) classifier to predict image categories based on pixel similarities.

🚀 Features
Custom RGBImage Class: Represents images using pixel data arrays.

Standard Image Processing:

Negate image colors

Convert to grayscale

Rotate image 180 degrees

Adjust brightness

Apply blurring filter

Premium Image Processing:

Chroma key replacement (green screen effect)

Sticker overlay on background images

Edge detection using convolution

KNN Image Classifier:

Fits labeled training images

Predicts categories of new images based on pixel similarity

🛠️ Tech Stack
Python 3

Numpy

Pillow (PIL)

📂 Project Structure
perl
Copy
Edit
.
├── main.py         # Full project code
├── knn_data/       # (Optional) Folder containing labeled images for KNN classification
├── img/            # Folder for testing and output images
│   ├── test_image_32x32.png
│   ├── exp/
│   └── out/
└── README.md       # Project documentation
📸 Example Functions
negate(): Inverts the color of every pixel.

grayscale(): Averages RGB values to create black-and-white images.

rotate_180(): Flips the image upside down and backward.

chroma_key(): Replaces a selected color (e.g., green screen) with background pixels.

predict(): Predicts the label of a new image using the k-nearest neighbors.

🔗 How to Run
Install dependencies:

bash
Copy
Edit
pip install numpy pillow
Clone this repository and navigate to it:

bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
You can run tests by importing the classes in a Python script or Jupyter Notebook, or use provided helper functions like img_read_helper() and img_save_helper().

📚 Sources
Developed based on standard image processing algorithms

Some dataset images are adapted from public domain repositories

✨ Future Enhancements
Expand KNN to allow different distance metrics (e.g., cosine similarity)

Build a web interface to upload images and classify them in real-time
