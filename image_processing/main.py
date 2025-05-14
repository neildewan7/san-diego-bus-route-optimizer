import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3

def img_read_helper(path):
    img = Image.open(path).convert("RGB")
    matrix = np.array(img).tolist()
    return RGBImage(matrix)

def img_save_helper(path, image):
    img_array = np.array(image.get_pixels())
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)

class RGBImage:
    def __init__(self, pixels):
        if not isinstance(pixels, list) or not pixels:
            raise TypeError()
        if not all(isinstance(pixel, list) for row in pixels for pixel in row):
            raise TypeError()
        if not all(len(row) == len(pixels[0]) for row in pixels):
            raise TypeError()
        if not all(len(pixel) == len(pixels[0][0]) for row in pixels for pixel in row):
            raise TypeError()
        if not all(0 <= intensity <= 255 for row in pixels for pixel in row for intensity in pixel):
            raise ValueError()
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        return [[[val for val in pixel] for pixel in row] for row in self.pixels]

    def copy(self):
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        if not (0 <= row < self.num_rows) or not (0 <= col < self.num_cols):
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        if not isinstance(new_color, tuple) or len(new_color) != 3:
            raise TypeError()
        if not all(0 <= c <= 255 for c in new_color):
            raise ValueError()
        self.pixels[row][col] = list(new_color)

class ImageProcessingTemplate:
    def __init__(self):
        self.cost = 0

    def get_cost(self):
        return self.cost

    def negate(self, image):
        img = image.copy()
        img.pixels = [[[255 - val for val in pixel] for pixel in row] for row in img.get_pixels()]
        return img

    def grayscale(self, image):
        img = image.copy()
        img.pixels = [[[sum(pixel) // 3] * 3 for pixel in row] for row in img.get_pixels()]
        return img

    def rotate_180(self, image):
        img = image.copy()
        img.pixels = [list(reversed(row)) for row in reversed(img.get_pixels())]
        return img

    def adjust_brightness(self, image, intensity):
        if not isinstance(intensity, int) or not (-255 <= intensity <= 255):
            raise ValueError()
        img = image.copy()
        img.pixels = [[[min(max(c + intensity, 0), 255) for c in pixel] for pixel in row] for row in img.get_pixels()]
        return img

    def blur(self, image):
        img = image.copy()
        pixels = img.get_pixels()
        new_pixels = []
        for i, row in enumerate(pixels):
            new_row = []
            for j, pixel in enumerate(row):
                neighbors = [pixels[x][y] for x in range(max(0, i-1), min(len(pixels), i+2)) for y in range(max(0, j-1), min(len(row), j+2))]
                avg_pixel = [sum(p[k] for p in neighbors) // len(neighbors) for k in range(3)]
                new_row.append(avg_pixel)
            new_pixels.append(new_row)
        img.pixels = new_pixels
        return img

class PremiumImageProcessing(ImageProcessingTemplate):
    def __init__(self):
        super().__init__()
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        if chroma_image.size() != background_image.size():
            raise ValueError()
        color = list(color)
        img = chroma_image.copy()
        for i in range(len(img.pixels)):
            for j in range(len(img.pixels[0])):
                if img.pixels[i][j] == color:
                    img.pixels[i][j] = background_image.get_pixels()[i][j]
        return img

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        if sticker_image.size()[0] + y_pos > background_image.size()[0] or sticker_image.size()[1] + x_pos > background_image.size()[1]:
            raise ValueError()
        background = background_image.copy()
        for i in range(sticker_image.size()[0]):
            for j in range(sticker_image.size()[1]):
                background.pixels[i + y_pos][j + x_pos] = sticker_image.get_pixels()[i][j]
        return background

    def edge_highlight(self, image):
        img = image.copy()
        avg = [[sum(pixel) // 3 for pixel in row] for row in img.get_pixels()]
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        for i in range(len(avg)):
            for j in range(len(avg[0])):
                value = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= i + dx < len(avg) and 0 <= j + dy < len(avg[0]):
                            value += avg[i+dx][j+dy] * kernel[dx+1][dy+1]
                value = min(max(value, 0), 255)
                img.pixels[i][j] = [value] * 3
        return img

class ImageKNNClassifier:
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def fit(self, data):
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, img1, img2):
        pixels1 = img1.get_pixels()
        pixels2 = img2.get_pixels()
        return (sum(sum(sum((a-b)**2 for a, b in zip(p1, p2)) for p1, p2 in zip(r1, r2)) for r1, r2 in zip(pixels1, pixels2))) ** 0.5

    def vote(self, labels):
        return max(set(labels), key=labels.count)

    def predict(self, img):
        dists = sorted((self.distance(img, data_img), label) for data_img, label in self.data)
        return self.vote([label for _, label in dists[:self.k_neighbors]])

def knn_tests(test_img_path):
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                data.append((img_read_helper(img_path), label))
    knn = ImageKNNClassifier(5)
    knn.fit(data)
    test_img = img_read_helper(test_img_path)
    return knn.predict(test_img)
