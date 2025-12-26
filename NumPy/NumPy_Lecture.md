# NumPy: Comprehensive Lecture Guide

## Introduction to NumPy

NumPy stands for **Numerical Python** and is one of the most fundamental libraries in Python for scientific computing. It provides support for:

- Large multidimensional arrays and matrices
- Mathematical functions to operate on these arrays
- High-performance computation and efficiency

### Why NumPy Matters

NumPy is essential for:

- Data science and machine learning
- Scientific computing
- Mathematical analysis
- Image processing
- Building blocks for other libraries (Pandas, Matplotlib, SciPy, Scikit-learn)

---

## Python Lists vs NumPy Arrays

### Python Lists

- Collection of heterogeneous data types
- Stored in non-contiguous memory locations
- Slow for mathematical operations (element-by-element type checking)
- More memory overhead

### NumPy Arrays

- Collection of homogeneous data types
- Stored in contiguous memory locations
- Vectorized operations (operates on entire arrays without explicit loops)
- Memory efficient
- Significantly faster computations

### Performance Comparison

```python
import time
import numpy as np

# Python list approach
list_data = list(range(1000000))
start = time.time()
result = list_data * 2
print("List time:", time.time() - start)

# NumPy approach
numpy_data = np.arange(1000000)
start = time.time()
result = numpy_data * 2
print("NumPy time:", time.time() - start)
```

**NumPy is typically 100-1000x faster for large arrays!**

---

## Installation and Import

```python
# Install NumPy (if not already installed)
# pip install numpy

# Import NumPy
import numpy as np
```

---

## Creating NumPy Arrays

### 1. From Python Lists

```python
# 1D array
arr1d = np.array([1, 2, 3, 4, 5])

# 2D array (matrix)
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Check data type
print(arr1d.dtype)  # int64
```

### 2. Using Creation Functions

```python
# Array of zeros
zeros = np.zeros((2, 3))
# [[0. 0. 0.]
#  [0. 0. 0.]]

# Array of ones
ones = np.ones((3, 4))

# Array with specific value
full = np.full((2, 2), 7)
# [[7 7]
#  [7 7]]

# Identity matrix
identity = np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Random numbers (0 to 1)
random = np.random.random((2, 3))

# Evenly spaced numbers
arange = np.arange(0, 10, 2)  # [0 2 4 6 8]

linspace = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]
```

---

## Array Attributes

### 1. **ndim** - Number of Dimensions

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.ndim)  # 2
```

### 2. **shape** - Size Along Each Dimension

```python
print(arr.shape)  # (2, 3) - 2 rows, 3 columns
print(arr.shape[0])  # 2 rows
print(arr.shape[1])  # 3 columns
```

### 3. **size** - Total Number of Elements

```python
print(arr.size)  # 6 (total elements)
```

### 4. **dtype** - Data Type of Elements

```python
print(arr.dtype)  # int64

# Specify dtype
arr_float = np.array([1, 2, 3], dtype=np.float32)
```

### 5. **T** - Transpose

```python
transposed = arr.T
# [[1 4]
#  [2 5]
#  [3 6]]
```

---

## Indexing and Slicing

### 1D Array Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5])

# Basic slicing [start:end:step]
print(arr[1:5:2])  # [1 3]
print(arr[:3])     # [0 1 2]
print(arr[2:])     # [2 3 4 5]
print(arr[::2])    # [0 2 4]

# Negative indexing
print(arr[-1])     # 5
print(arr[:-1])    # [0 1 2 3 4]
```

### 2D Array Indexing

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Single element
print(arr[0, 0])   # 1
print(arr[1, 2])   # 6

# Row slicing
print(arr[0, :])   # [1 2 3]

# Column slicing
print(arr[:, 1])   # [2 5]

# 2D slicing
print(arr[:2, 1:3])
# [[2 3]
#  [5 6]]
```

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[arr > 3])  # [4 5]

# Create boolean mask
mask = arr > 2
print(arr[mask])  # [3 4 5]
```

---

## Arithmetic Operations

### Element-wise Operations

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Basic operations
print(a + b)      # Element-wise addition
print(a - b)      # Element-wise subtraction
print(a * b)      # Element-wise multiplication
print(a / b)      # Element-wise division
print(a ** 2)     # Element-wise power
print(a % 3)      # Element-wise modulus
```

### Operations with Scalars

```python
arr = np.array([1, 2, 3])

print(arr + 5)    # [6 7 8]
print(arr * 2)    # [2 4 6]
print(arr / 2)    # [0.5 1.  1.5]
```

### Mathematical Functions

```python
arr = np.array([0, np.pi/2, np.pi])

print(np.sin(arr))    # Sine
print(np.cos(arr))    # Cosine
print(np.sqrt(arr))   # Square root
print(np.exp(arr))    # Exponential
print(np.log(arr))    # Natural logarithm
print(np.abs(arr))    # Absolute value
```

---

## Aggregate Functions

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Sum all elements
print(np.sum(arr))    # 21

# Sum along axis
print(np.sum(arr, axis=0))  # [5 7 9] (sum columns)
print(np.sum(arr, axis=1))  # [6 15] (sum rows)

# Other aggregates
print(np.mean(arr))   # 3.5
print(np.std(arr))    # Standard deviation
print(np.min(arr))    # 1
print(np.max(arr))    # 6
print(np.median(arr)) # Median
```

### Finding Min/Max Index

```python
arr = np.array([5, 2, 8, 1, 9])
print(np.argmin(arr))  # 3 (index of minimum)
print(np.argmax(arr))  # 4 (index of maximum)
```

---

## Broadcasting

Broadcasting allows NumPy to work with arrays of different shapes during arithmetic operations.

### Broadcasting Rules

1. If arrays have different ranks, prepend 1s to the smaller array
2. Arrays are compatible if dimensions are equal or one is 1
3. After broadcasting, each array acts as if it had the shape of element-wise maximum

### Examples

```python
# Adding 1D array to 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
vec = np.array([1, 0, 1])

result = arr + vec
# [[2 2 4]
#  [5 5 7]]

# The vector is broadcast across each row!

# Adding scalar
result = arr + 5
# [[6  7  8]
#  [9 10 11]]
```

### Practical Broadcasting Examples

```python
# Outer product
v = np.array([1, 2, 3])    # shape (3,)
w = np.array([4, 5])        # shape (2,)

outer = np.reshape(v, (3, 1)) * w
# [[4  5]
#  [8 10]
#  [12 15]]

# Normalize data (subtract mean)
data = np.array([[1, 2, 3],
                 [4, 5, 6]])
mean = np.mean(data, axis=0)  # [2.5 3.5 4.5]
normalized = data - mean
```

---

## Reshaping Arrays

### Reshape

```python
arr = np.arange(12)  # [0 1 2 3 ... 11]

reshaped = arr.reshape(3, 4)
# [[0  1  2  3]
#  [4  5  6  7]
#  [8  9 10 11]]

# Auto-compute one dimension with -1
reshaped = arr.reshape(3, -1)  # NumPy calculates the -1 dimension

reshaped = arr.reshape(-1, 6)
```

### Flatten vs Ravel

```python
arr = np.array([[1, 2], [3, 4]])

# Flatten - returns a copy
flat = arr.flatten()  # [1 2 3 4]
flat[0] = 99
print(arr)  # Original unchanged

# Ravel - returns a view (reference)
flat_view = arr.ravel()
flat_view[0] = 99
print(arr)  # Original changed!
```

### Transpose

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

transposed = arr.T
# [[1 4]
#  [2 5]
#  [3 6]]

# For 1D arrays, transpose does nothing
v = np.array([1, 2, 3])
print(v.T)  # [1 2 3]
```

### Expand/Squeeze Dimensions

```python
arr = np.array([1, 2, 3])  # shape (3,)

# Add dimension
expanded = np.expand_dims(arr, axis=0)  # shape (1, 3)
expanded = np.expand_dims(arr, axis=1)  # shape (3, 1)

# Remove dimensions of size 1
arr2d = np.array([[1, 2, 3]])  # shape (1, 3)
squeezed = np.squeeze(arr2d)   # shape (3,)
```

---

## Concatenation and Stacking

### Concatenate

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenate along rows (axis=0)
result = np.concatenate((a, b), axis=0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Concatenate along columns (axis=1)
result = np.concatenate((a, b), axis=1)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### Stack

```python
# Vertical stack (vstack) - like axis=0
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
v_stacked = np.vstack((a, b))
# [[1 2 3]
#  [4 5 6]]

# Horizontal stack (hstack) - like axis=1
h_stacked = np.hstack((a, b))
# [1 2 3 4 5 6]

# Depth stack (dstack) - along axis=2
d_stacked = np.dstack((a, b))
```

---

## Sorting Arrays

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sort array (returns sorted copy)
sorted_arr = np.sort(arr)
# [1 1 2 3 4 5 6 9]

# Get indices that would sort the array
indices = np.argsort(arr)
# [1 3 6 0 2 4 7 5]

# 2D sorting
arr2d = np.array([[5, 3],
                  [4, 1]])
sorted_2d = np.sort(arr2d, axis=1)  # Sort along rows
```

---

## Matrix Operations

### Matrix Multiplication (Dot Product)

```python
# Vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = v1.dot(v2)  # 32
# 1*4 + 2*5 + 3*6 = 32

# Matrix-Vector multiplication
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([1, 0, 1])

result = matrix.dot(vector)  # [4 10]

# Matrix-Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

result = A.dot(B)
# [[19 22]
#  [43 50]]
```

### Element-wise vs Matrix Multiplication

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A * B)         # Element-wise: [[5 12] [21 32]]
print(A.dot(B))      # Matrix multiplication
print(np.dot(A, B))  # Same as A.dot(B)
```

---

## Comparison Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 5, 2])

# Comparison returns boolean array
print(a == b)  # [True  True False  True]
print(a != b)  # [False False True False]
print(a < b)   # [False False True False]
print(a > b)   # [False False False True]
print(a <= b)  # [True  True False True]
print(a >= b)  # [True  True False True]

# Use with conditionals
print(np.all(a < 5))    # True - all elements < 5?
print(np.any(a > 3))    # True - any element > 3?
```

---

## Random Number Generation

### Basic Random Functions

```python
# Random numbers between 0 and 1
rand_vals = np.random.random(5)  # 1D array of 5 random values
rand_matrix = np.random.random((3, 3))  # 3x3 matrix

# Random integers
int_array = np.random.randint(0, 10, size=5)  # 5 random ints between 0-10

# Normal distribution (mean=0, std=1)
normal = np.random.normal(0, 1, size=100)

# Custom distribution
custom = np.random.normal(loc=100, scale=15, size=50)  # mean=100, std=15

# Uniform distribution
uniform = np.random.uniform(5, 25, size=(2, 3))  # between 5 and 25

# Random choice from array
arr = np.array([1, 2, 3, 4, 5])
choices = np.random.choice(arr, size=10, replace=True)
```

### Seeding for Reproducibility

```python
np.random.seed(42)
random1 = np.random.random(5)

np.random.seed(42)
random2 = np.random.random(5)

print(np.array_equal(random1, random2))  # True - same results!
```

---

## Working with Images

Images are stored as NumPy arrays where each pixel has values 0-255.

### Image Basics

```python
# For grayscale images: 2D array (height, width)
# For color images: 3D array (height, width, 3) for RGB channels

from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image

# Create random image
image = np.random.rand(100, 100, 3) * 255
image = image.astype(np.uint8)

print(f"Image shape: {image.shape}")  # (height, width, channels)
print(f"Image dtype: {image.dtype}")  # uint8
```

### Loading and Displaying Images

```python
# Load image from file
from PIL import Image

img = Image.open('path/to/image.jpg')
img_array = np.array(img)

# Display image
plt.imshow(img_array)
plt.title('Original Image')
plt.axis('off')
plt.show()
```

### Image Manipulation Operations

```python
# Flip/Mirror operations
flipped_horizontal = np.flip(image, axis=1)  # Flip left-right
flipped_vertical = np.flip(image, axis=0)    # Flip top-bottom

# Rotate (90, 180, 270 degrees)
rotated_90 = np.rot90(image)

# Transpose (swap rows and columns)
transposed = image.T

# Crop image (select region)
cropped = image[50:150, 50:150, :]  # Extract 100x100 region

# Resize (using PIL)
resized_img = Image.fromarray(image)
resized_img = resized_img.resize((200, 200))
resized_array = np.array(resized_img)
```

### Color Space Operations

```python
# Normalize pixel values to 0-1
normalized = image / 255.0

# Convert to grayscale (average of RGB channels)
grayscale = np.mean(image, axis=2)

# Extract color channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Remove color channels (set to 0)
image_no_red = image.copy()
image_no_red[:, :, 0] = 0  # Remove red channel

# Adjust brightness
brightened = np.clip(image * 1.5, 0, 255).astype(np.uint8)
darkened = np.clip(image * 0.5, 0, 255).astype(np.uint8)

# Tint image (apply color multiplier)
tinted = image * np.array([1.0, 0.9, 0.8])  # Reduce green and blue
tinted = np.clip(tinted, 0, 255).astype(np.uint8)
```

### Image Filtering and Effects

```python
from scipy import ndimage

# Blur effect (Gaussian filter)
blurred = ndimage.gaussian_filter(image, sigma=2)

# Edge detection (Sobel filter)
edges = ndimage.sobel(grayscale)

# Invert colors
inverted = 255 - image

# Threshold (convert to binary - black and white)
threshold = (grayscale > 128).astype(np.uint8) * 255

# Histogram equalization (enhance contrast)
histogram = np.histogram(grayscale, bins=256)

# Blur specific channels
image_blurred = image.copy()
image_blurred[:, :, 0] = ndimage.gaussian_filter(image[:, :, 0], sigma=2)
```

### Practical Image Processing Example

```python
# Load and process image
img = Image.open('photo.jpg')
img_array = np.array(img)

# Convert to grayscale
gray = np.mean(img_array, axis=2)

# Apply threshold
binary = (gray > np.mean(gray)).astype(np.uint8) * 255

# Resize
resized = Image.fromarray(gray).resize((256, 256))
resized_array = np.array(resized)

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_array)
axes[0].set_title('Original')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[2].imshow(binary, cmap='gray')
axes[2].set_title('Binary')
plt.show()
```

---

## Working with Audio

Audio data is typically stored as NumPy arrays where values represent sound wave amplitudes at different time points.

### Audio Basics

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa

# Audio properties:
# - Sample rate: Hz (samples per second, typically 44100, 48000)
# - Amplitude: Magnitude of sound wave
# - Duration: Length in seconds

# Create synthetic audio
sample_rate = 44100  # Hz
duration = 2  # seconds
t = np.linspace(0, duration, sample_rate * duration)

# Generate sine wave (frequency = 440 Hz, A note)
frequency = 440
audio_data = np.sin(2 * np.pi * frequency * t) * 0.3  # Amplitude 0.3
```

### Loading and Saving Audio

```python
# Load audio file
from scipy.io import wavfile
sample_rate, audio_array = wavfile.read('audio.wav')

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio shape: {audio_array.shape}")
print(f"Duration: {len(audio_array) / sample_rate} seconds")

# Save audio file
output_audio = np.array(audio_data * 32767, dtype=np.int16)  # Convert to 16-bit
wavfile.write('output.wav', sample_rate, output_audio)

# Using librosa (more features)
audio, sr = librosa.load('audio.wav')
# audio is normalized between -1 and 1
```

### Audio Visualization

```python
# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(t, audio_data)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.grid(True)
plt.show()

# Spectrogram (frequency vs time)
plt.specgram(audio_data, Fs=sample_rate, cmap='viridis')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.colorbar(label='Intensity')
plt.show()
```

### Audio Processing

```python
# Normalize audio (keep values between -1 and 1)
max_val = np.abs(audio_data).max()
normalized = audio_data / max_val

# Change volume (amplitude scaling)
louder = audio_data * 2.0
quieter = audio_data * 0.5

# Trim silence (remove low amplitude sections)
threshold = 0.01
mask = np.abs(audio_data) > threshold
trimmed = audio_data[mask]

# Reverse audio
reversed_audio = audio_data[::-1]

# Speed up/Slow down
speed_factor = 1.5
# Create new time array at different intervals
sped_up = audio_data[::int(1/speed_factor)]

# Fade in/out effects
fade_in = np.linspace(0, 1, len(audio_data) // 4)
fade_out = np.linspace(1, 0, len(audio_data) // 4)
audio_faded = audio_data.copy()
audio_faded[:len(fade_in)] *= fade_in
audio_faded[-len(fade_out):] *= fade_out
```

### Frequency Analysis

```python
# Fast Fourier Transform (FFT)
fft = np.fft.fft(audio_data)
frequencies = np.fft.fftfreq(len(audio_data), 1/sample_rate)

# Get magnitude spectrum
magnitude = np.abs(fft)

# Plot frequency spectrum
plt.figure(figsize=(12, 4))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.xlim(0, 5000)  # Show up to 5kHz
plt.show()

# Find dominant frequency
dominant_freq = frequencies[np.argmax(magnitude)]
print(f"Dominant frequency: {dominant_freq} Hz")

# Create tone at specific frequencies
def create_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, sample_rate * duration)
    return np.sin(2 * np.pi * frequency * t)

# Mix multiple frequencies
c_note = create_tone(262, 1, sample_rate)
e_note = create_tone(330, 1, sample_rate)
g_note = create_tone(392, 1, sample_rate)
chord = c_note + e_note + g_note
chord /= np.max(np.abs(chord))  # Normalize
```

### Audio Feature Extraction

```python
# Extract features using librosa
import librosa

audio, sr = librosa.load('audio.wav')

# Zero crossing rate (changes in sign)
zcr = librosa.feature.zero_crossing_rate(audio)

# Spectral centroid (center of mass of spectrum)
spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

# MFCC (Mel-frequency cepstral coefficients)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Energy
energy = np.sqrt(np.sum(audio**2))

print(f"Zero Crossing Rate shape: {zcr.shape}")
print(f"Spectral Centroid shape: {spectral_centroid.shape}")
print(f"MFCC shape: {mfcc.shape}")
print(f"Energy: {energy}")
```

---

## Natural Language Processing (NLP) Basics

NLP involves processing text data. NumPy helps with numerical operations on text-derived features.

### Text to Numbers

```python
import numpy as np

# Tokenization - split text into words
text = "Hello world! This is a test."
tokens = text.lower().split()
print(tokens)  # ['hello', 'world', 'this', 'is', 'a', 'test']

# Create vocabulary (unique words)
vocabulary = list(set(tokens))
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(word_to_idx)
```

### Bag of Words (BoW)

```python
# Bag of Words - count word occurrences
def bow_vector(text, vocab):
    tokens = text.lower().split()
    bow = np.zeros(len(vocab))
    for token in tokens:
        if token in word_to_idx:
            bow[word_to_idx[token]] += 1
    return bow

sentence1 = "the cat sat on the mat"
sentence2 = "the dog sat on the log"

# Create vocabulary
all_words = set(sentence1.split() + sentence2.split())
vocab = list(all_words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Convert to BoW vectors
vec1 = bow_vector(sentence1, vocab)
vec2 = bow_vector(sentence2, vocab)

print("Sentence 1 BoW:", vec1)
print("Sentence 2 BoW:", vec2)
```

### Similarity Measures

```python
# Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.sqrt(np.sum(vec1**2))
    magnitude2 = np.sqrt(np.sum(vec2**2))
    return dot_product / (magnitude1 * magnitude2)

# Euclidean Distance
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

similarity = cosine_similarity(vec1, vec2)
distance = euclidean_distance(vec1, vec2)

print(f"Cosine Similarity: {similarity:.3f}")
print(f"Euclidean Distance: {distance:.3f}")
```

### Word Embeddings (Simple Example)

```python
# One-hot encoding - represent each word as a vector
def one_hot_encode(word, vocab):
    vector = np.zeros(len(vocab))
    vector[word_to_idx[word]] = 1
    return vector

# Create one-hot vectors
word_vectors = {}
for word in vocab:
    word_vectors[word] = one_hot_encode(word, vocab)

print("One-hot encoding for 'cat':")
print(word_vectors['cat'])
```

### Text Statistics with NumPy

```python
text = "Hello world! This is a simple text. Text processing is fun!"
words = text.lower().split()
word_lengths = np.array([len(word.strip('.,!?')) for word in words])

# Statistics
print(f"Average word length: {np.mean(word_lengths):.2f}")
print(f"Max word length: {np.max(word_lengths)}")
print(f"Min word length: {np.min(word_lengths)}")
print(f"Std deviation: {np.std(word_lengths):.2f}")

# Word frequency distribution
from collections import Counter
freq = Counter(words)
frequencies = np.array(list(freq.values()))

print(f"Most common word frequency: {np.max(frequencies)}")
print(f"Average frequency: {np.mean(frequencies):.2f}")
```

### TF-IDF (Term Frequency - Inverse Document Frequency)

```python
def tf_idf(documents):
    """
    documents: list of strings (each string is a document)
    Returns: TF-IDF matrix (documents x vocabulary)
    """
    # Tokenize
    all_words = set()
    tokenized = []
    for doc in documents:
        tokens = doc.lower().split()
        tokenized.append(tokens)
        all_words.update(tokens)

    vocab = list(all_words)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Calculate TF (Term Frequency)
    tf_matrix = np.zeros((len(documents), len(vocab)))
    for doc_idx, tokens in enumerate(tokenized):
        for token in tokens:
            tf_matrix[doc_idx, word_to_idx[token]] += 1
        tf_matrix[doc_idx] /= len(tokens)  # Normalize by doc length

    # Calculate IDF (Inverse Document Frequency)
    idf = np.zeros(len(vocab))
    for word_idx in range(len(vocab)):
        # Count documents containing this word
        docs_with_word = np.sum(tf_matrix[:, word_idx] > 0)
        idf[word_idx] = np.log(len(documents) / (1 + docs_with_word))

    # TF-IDF = TF * IDF
    tfidf = tf_matrix * idf
    return tfidf, vocab

# Example
documents = [
    "the cat sat on the mat",
    "the dog played in the park",
    "the cat and dog are friends"
]

tfidf_matrix, vocab = tf_idf(documents)
print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Vocabulary:", vocab)
```

### Basic Sentiment from Word Counts

```python
# Simple sentiment analysis using word polarity
positive_words = {'good', 'great', 'excellent', 'love', 'happy', 'best'}
negative_words = {'bad', 'terrible', 'hate', 'sad', 'worst', 'poor'}

def simple_sentiment(text):
    words = set(text.lower().split())

    positive_count = len(words & positive_words)
    negative_count = len(words & negative_words)

    sentiment_score = positive_count - negative_count

    return sentiment_score

# Test
texts = [
    "I love this! It's great and excellent!",
    "This is terrible and bad",
    "It's good but also has some bad parts"
]

for text in texts:
    score = simple_sentiment(text)
    print(f"Text: '{text}'")
    print(f"Sentiment Score: {score}\n")
```

---

---

## Common Operations Summary

| Operation    | Code                         |
| ------------ | ---------------------------- |
| Create array | `np.array([1,2,3])`          |
| Zeros        | `np.zeros((3,3))`            |
| Ones         | `np.ones((2,4))`             |
| Random       | `np.random.random((2,3))`    |
| Range        | `np.arange(0, 10, 2)`        |
| Linspace     | `np.linspace(0, 1, 5)`       |
| Shape        | `arr.shape`                  |
| Size         | `arr.size`                   |
| Reshape      | `arr.reshape(2, 3)`          |
| Flatten      | `arr.flatten()`              |
| Transpose    | `arr.T`                      |
| Sum          | `np.sum(arr)`                |
| Mean         | `np.mean(arr)`               |
| Std          | `np.std(arr)`                |
| Min/Max      | `np.min(arr)`, `np.max(arr)` |
| Sort         | `np.sort(arr)`               |
| Dot product  | `a.dot(b)`                   |

---

## Best Practices

1. **Use NumPy for numerical operations** - It's much faster than Python loops
2. **Leverage broadcasting** - Avoid explicit loops when possible
3. **Specify dtypes** - Control memory usage and precision
4. **Use vectorized operations** - Apply functions to entire arrays
5. **Be aware of views vs copies** - Know when you're modifying original arrays
6. **Use proper data structures** - 1D for vectors, 2D for matrices, 3D for images

---

## Common Pitfalls

```python
# Wrong - modifying original with ravel()
arr = np.array([[1, 2], [3, 4]])
flat = arr.ravel()
flat[0] = 99  # Changes original!

# Right - use flatten() for independent copy
flat = arr.flatten()
flat[0] = 99  # Original unchanged

# Wrong - list vs array operations
x = np.array([1, 2, 3])
y = [1, 2, 3]
# x + x is fast, but x + y requires conversion

# Right - keep data in arrays
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
z = x + y  # Fast operation
```

---

## Conclusion

NumPy is essential for:

- **Data Science** - Handle large datasets efficiently
- **Scientific Computing** - Mathematical operations at scale
- **Machine Learning** - Foundation for TensorFlow, PyTorch, Scikit-learn
- **Image Processing** - Manipulate image data
- **Financial Analysis** - Process financial data

**Master NumPy to become proficient in Python data science!**

---

## Resources

- [NumPy Official Documentation](https://numpy.org/doc/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Tutorials](https://numpy.org/doc/stable/user/basics.html)
