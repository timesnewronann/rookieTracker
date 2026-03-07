# ShotTracker Learning Log

This document tracks the concepts learned while building the ShotTracker project from scratch.

The goal of this project is to gradually build a basketball shot tracking system similar to HomeCourt by first understanding the fundamental building blocks of computer vision systems.

---

# Version 0 — Video Foundations

Goal: Understand how video files are processed frame-by-frame using OpenCV.

Before detecting a basketball, we must understand how video is represented and processed in code.

---

# Lesson 1 — Opening a Video File

### Concept

OpenCV reads videos using the `VideoCapture` object.

```python
cap = cv.VideoCapture(video_path)
```

This creates a video stream object that allows us to read frames sequentially.

---

### Checking if the video opened successfully

```python
if not cap.isOpened():
    print("Error: Could not open video.")
```

Why this matters:

If the file path is wrong or the video cannot be decoded, the program should stop early.

---

# Lesson 2 — Reading Frames

Video is simply a **sequence of images**.

We read one frame at a time using:

```python
ret, frame = cap.read()
```

### What these values mean

`ret`

Boolean value indicating whether the frame was successfully read.

```
True  → frame read successfully
False → no frame available
```

`frame`

A NumPy array representing the image pixels.

Example shape:

```
(1080, 1920, 3)
```

Meaning:

```
height = 1080 pixels
width  = 1920 pixels
channels = 3 (BGR color channels)
```

---

### Why the loop stops

When the video ends:

```
ret = False
frame = None
```

This triggers:

```python
if not ret:
    break
```

Which exits the loop cleanly.

---

# Lesson 3 — Displaying Frames

Frames can be displayed using:

```python
cv.imshow("frame", frame)
```

This opens a window and renders the image.

---

# Lesson 4 — Keyboard Control

To allow the user to quit the program:

```python
cv.waitKey(1)
```

This waits 1 millisecond for keyboard input.

We exit if the user presses **q**:

```python
if cv.waitKey(1) == ord('q'):
    break
```

---

# Lesson 5 — Cleaning Up Resources

When finished reading the video we must release resources.

```python
cap.release()
cv.destroyAllWindows()
```

Why this matters:

- releases the video file
- closes GUI windows
- prevents memory leaks

---

# Lesson 6 — Video Metadata

OpenCV allows us to inspect properties of the video.

These are accessed with:

```python
cap.get(property)
```

Important properties:

### Width

```python
cv.CAP_PROP_FRAME_WIDTH
```

### Height

```python
cv.CAP_PROP_FRAME_HEIGHT
```

### Frames Per Second (FPS)

```python
cv.CAP_PROP_FPS
```

### Total Frame Count

```python
cv.CAP_PROP_FRAME_COUNT
```

---

# Calculating Video Duration

Duration is derived from:

```
duration = frame_count / fps
```

Example output from our test clip:

```
Video Width: 3840
Video Height: 2160
FPS: 23
Frame Count: 1683
Duration: 73.17 seconds
```

---

# Why Metadata Matters for Shot Tracking

Later versions of ShotTracker will rely heavily on this information.

Examples:

### Shot release timing

```
Release frame: 210
Hoop entry frame: 245
Flight time: 35 frames
```

Convert to seconds:

```
35 / FPS
```

Which gives the **ball flight duration**.

This will eventually allow analysis of:

- shot arc
- shot speed
- shot timing
- rhythm

---

# Key Concepts Learned

• Video = sequence of frames
• Each frame is an image (NumPy array)
• Frames are processed sequentially
• Metadata describes the structure of the video
• Computer vision pipelines operate frame-by-frame

---

# Current Project Version

Version 0.1 — Video playback and metadata inspection

Capabilities:

✓ Open video file
✓ Read frames sequentially
✓ Display frames
✓ Exit with keyboard
✓ Inspect video metadata

---

# Next Version

Version 1 — Basketball Detection (Color Segmentation)

Goal:

Detect the basketball in a simple controlled clip using color masking.

Concepts we will learn:

• HSV color space
• Image masking
• Contours
• Extracting object center points

# Lesson - Frame Downscaling

High-resolution video significantly increases computation

Example:
4K Resolution
3840 x 2160 = 8,294,400 pixels per frame
If processing 23 frames per second:

≈ 191 million pixels processed every second.

To improve performance, many computer vision systems downscale frames before proccessing.

Example:

```python
frame = cv.resize(frame, (1280, 720))
```

# Lesson - HSV Color Space

HSV for detection
HSV separates color from brightness

H = Hue
S = Saturation
V = Value (brightness)

OpenCV images are normally loaded in BGR format, which mixes color and brightness.

This makes color detection unreliable because lighting changes the BGR values significantly.

HSV separates the actual color (Hue) from brightness (Value)

Hue represents the color itself

Isolate for the orange basketball with this code

```python
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, lower_orange, upper_orange)
```

Produces a binary mask
white = possible basketball pixels
black = everything else

# Lesson - False Positive in Color Detection

Color masking does not detect "a basketball."

It only detects pixels within a chosen color range.

This means other orange objects may also be detected.

Examples from current video:

- orange shorts
- shot clock glow
- reflections or lighting
- another basketball
- other orange objects in the scene

Incorrect Detections == false positives

Color is not enough for reliable basketball detection

We will add more filters to improve detection later

- position filter (region of interest)
- size filter
- shape filter
- motion filter
- tracking from previous frame

A simple detector often gives noisy results first
It will improve by layering constraints

# Lesson — What a Mask Looks Like

A color mask is a binary image.

It is not a normal color image.

Pixel meaning:

- white (255) = pixel matches the target color range
- black (0) = pixel does not match

For basketball detection:

- orange parts of the frame become white
- everything else becomes black

This makes the image much easier to analyze because the system only needs to focus on matching regions instead of all original pixel values.

# Lesson - Common Loop Bugs in OpenCV

Two common bugs appeared while building the mask display version.

## Bug 1 - Releasing the capture inside the loop

If `cap.release()` is called inside the frame-processing loop, the video stream closes too early.

Then the next call to:

```python
ret, frame = cap.read()
```

