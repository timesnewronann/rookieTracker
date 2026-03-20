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

# Lesson 7 - Frame Downscaling

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

# Lesson 8 - HSV Color Space

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

# Lesson 9 - False Positive in Color Detection

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

# Lesson 10 — What a Mask Looks Like

A color mask is a binary image.

It is not a normal color image.

Pixel meaning:

- white (255) = pixel matches the target color range
- black (0) = pixel does not match

For basketball detection:

- orange parts of the frame become white
- everything else becomes black

This makes the image much easier to analyze because the system only needs to focus on matching regions instead of all original pixel values.

# Lesson 11 - Common Loop Bugs in OpenCV

Two common bugs appeared while building the mask display version.

## Bug 1 - Releasing the capture inside the loop

If `cap.release()` is called inside the frame-processing loop, the video stream closes too early.

Then the next call to:

```python
ret, frame = cap.read()
```

# Lesson 12 - Contours

After creating a color mask, we still do not know which object is the basketball.

The mask may contain multiple white regions representing different orange objects.

Examples:

- basketball
- shorts
- shot clock
- reflections

To separate these objects we use **contours**.

Contours represent the outlines of connected white regions in a binary mask.

Example:

```python
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
```

# Lesson 13 - Filtering Contours by Size

After detecting contours from a mask, many of them may represent noise.

Examples:

- shot clock pixels
- reflections
- compression artificats

A simple way to remove these is by filtering contours based on area.

OpenCV provides:

```python
cv.contourArea(contour)
```

# Lesson 14 - Threshold Tradeoffs

Contour area filtering helps remove noise, but the threshold must be chosen carefully.

Examples:

```python
if area < 300:
    continue
```

# Lesson 15 - Area Filtering Is Not Enough

Filtering contours by area removes small noise
This doesn't gurantee that the remaining object is the basketball.

In the video, larger orange objects such as:

- the shot clock
- the cone and pole

Still pass the size filter

Contour area alone is not enough to identify the basketball.

To improve detection, we need additional filters such as:

- shape
- position
- motion

# Lesson 16 - Bounding Rectangles

When we detect a contour, draw a bounding rectangle around it

OpenCV function:

```python
x, y, w, h = cv.boundingRect(contour)
```

# Lesson 17 - Debug filters one at a Time

Filter may be rejecting the object before drawing happens.

Debugging Strategy:

1. Remove the newest filter
2. Confirm the object draws again
3. Print intermediate values
4. Re-Add the filter carefully

Makes it easier to identify whether the issue comes from:

- masking
- contour detection
- shape threshold
- drawing location

# Lesson 18 - Debugging Fast Video is difficult

When drawings do not appear clearly on video output, the problem may be visual inspection rather than the drawing code.

Common causes:

- video moves too fast
- detected objects are very small
- annotations are too thin
- the user is looking at the wrong output window

A strong debugging technique is to:

- pause on each frame with `cv.waitKey(0)`
- save an annotated frame with `cv.imwrite(...)`
- increase line thickness
- add text labels to detected objects

# Lesson 19 - A Bad Contour == A Bad Mask

If contour measurements mostly describe poles, strips, or small fragments, the issue is often earlier in the pipeline.

Bad contour output usually means:

- the color mask is too noisy
- the target object is fragmented
- the HSV range is not isolating the object well

This means contour filtering cannot fully solve the problem unless the mask is improved first.

# Lesson 20 — Debugging Is Easier with Display Toggles

When a computer vision pipeline has multiple outputs, it can be hard to understand what is happening if all views are shown at once.

A useful debugging pattern is to add display toggles.

Example modes:

- annotated frame
- binary mask
- masked color result

This allows the developer to inspect one stage of the pipeline at a time.

This is especially helpful when trying to answer questions like:

- Is the mask correct?
- Are contours being found?
- Are bounding boxes actually being drawn?
- Is the issue in detection or just in visualization?

# Lesson 21 — Region of Interest (ROI)

When a detector searches the entire frame, it often finds many irrelevant objects.

A Region of Interest (ROI) limits processing to only the part of the image where the target is likely to appear.

Example:

- ignore the upper wall
- ignore the shot clock
- ignore distant background objects

This improves detection by reducing false positives before contour filtering even begins.

ROI is one of the most practical ways to improve a classical computer vision pipeline.

# Lesson 22 — Static ROI vs Dynamic ROI

A Region of Interest (ROI) can be either fixed or dynamic.

## Static ROI

A fixed area of the frame that never moves.

Use case:

- controlled video
- known shooting area
- early debugging

## Dynamic ROI

A moving search region centered around the previously detected ball position.

Example idea:

1. detect the ball
2. save its center
3. search nearby in the next frame
4. update the center again

Dynamic ROI is useful because it reduces false positives and speeds up processing.

However, it depends on having a reliable initial detection.

If the first detection is wrong, the tracker may follow the wrong object.

# Lesson 23 — Why ROI Comes Before Smarter Filtering

When false positives appear across the whole frame, the best next step is often to reduce the search space.

A Region of Interest (ROI) helps by limiting detection to the area where the basketball is actually likely to appear.

Benefits:

- fewer false positives
- cleaner masks
- fewer contours to compare
- simpler debugging

For this project, a static ROI should be added before more advanced filtering or dynamic tracking.

Important coordinate rule:

If a contour is found inside the ROI at `(x, y)`, its position in the full frame is:

```python
full_x = roi_x1 + x
full_y = roi_y1 + y
```

# Lesson 25 — Separate Detection Space from Display Space

When using an ROI, detection happens inside the cropped image, but annotations are usually drawn on the full frame.

This creates two coordinate systems:

## Detection space

The cropped ROI

## Display space

The full frame

Contours and bounding boxes found inside the ROI use ROI-local coordinates.

To draw them correctly on the full frame, the ROI offset must be added:

```python
full_x = roi_x1 + x
full_y = roi_y1 + y
```

# Lesson 26 — ROI Changes Detection, Not Visualization

When using a Region of Interest (ROI):

- Detection happens ONLY inside the ROI
- Mask and contours are computed from the ROI
- Coordinates from the ROI must be translated back to the full frame

Key idea:

ROI reduces the search space, which improves detection quality by removing irrelevant regions.

Pipeline:

full frame → crop ROI → HSV → mask → contours → offset → draw on full frame

# Lesson 27 — ROI Reduces Search Space, But Does Not Identify the Object

A Region of Interest (ROI) helps remove irrelevant parts of the frame, but it does not automatically separate the basketball from other nearby objects.

In this project, both the basketball and the orange shorts stripe can still exist inside the ROI.

This means ROI is only a coarse filter.

It answers:

- where to search

But it does not answer:

- which object is the basketball

To solve that, additional filters are needed, such as:

- shape
- circularity
- motion
- tracking consistency

# Lesson 28 — Circularity

Circularity measures how close a contour is to a circle.

Formula:

```python
circularity = 4 * math.pi * area / (perimeter * perimeter)
```

# Lesson 29 — A Strong Filter Cannot Fix a Weak Contour

If area, aspect ratio, and circularity all fail to isolate the basketball, the issue may not be the filters themselves.

The issue may be that the ball is not forming a strong contour in the mask.

In that case, improving the mask is more important than adding stricter filters.

A common solution is morphology, especially closing:

```python
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
```

# Lesson 30 — Better Debugging Makes Better Tuning

When multiple candidate contours survive filtering, it can still be hard to understand which one is the most likely ball.

A useful debugging improvement is to draw:

- bounding box
- center point
- aspect ratio
- circularity

This makes it easier to compare candidates visually and decide which filters need tuning next.

# Lesson 31 — HSV Tuning Is Often More Important Than New Filters

If the detector consistently finds the wrong colored object, the issue may be the color range itself.

A broad HSV range can capture nearby colors such as yellow or washed-out orange.

To target a deeper basketball orange, we can:

- reduce the hue range
- increase minimum saturation
- increase minimum value

Example tuning idea:

```python
lower_orange = np.array([5, 140, 120])
upper_orange = np.array([18, 255, 255])
```

# Lesson 32 — Parameter Tuning Often Overshoots

When tuning HSV values, it is common to overshoot.

Example:

- broad range → too many false positives
- narrow range → target disappears

This does not mean HSV tuning failed. It means the range needs to be adjusted toward the middle.

A good tuning strategy is:

1. make one small change
2. test the result
3. compare what improved and what was lost
4. adjust gradually

# Lesson 33 — Measure the Real Pixel Values Before Guessing

When color thresholds are hard to tune, a better approach is to inspect the actual color values of the target object.

Instead of guessing HSV ranges, we can sample pixels directly from the frame.

Best practice:

- pause on a frame where the object is visible
- click on the object
- print BGR and HSV values
- average a small patch instead of using only one pixel

This makes threshold tuning data-driven instead of guess-based.

# Lesson 34 — Python Scope Can Break Debug Tools

A variable defined outside a function is global.

Example:

```python
clicked_frame = None
```

# Lesson 35 — Use Real Sampled HSV Values to Guide Thresholds

A sampled basketball patch produced approximately:

```python
H ≈ 4.6
S ≈ 127
V ≈ 134
```

# Lesson 36 — Once Detection Works, Do Not Expand the Search Space Too Quickly

When the target object is finally detected reliably, the next step is usually not to make the ROI bigger.

A larger ROI can reintroduce false positives and reduce stability.

Better question:

- what is the smallest ROI that still contains the ball during the important motion?

Smaller, correct search spaces usually improve:

- speed
- stability
- precision

# Lesson 38 — Tracking Requires One Best Candidate Per Frame

A detector may find multiple valid contours in one frame.

If all of them are appended to the path, the trail becomes a mixture of multiple objects instead of one tracked ball.

A tracker should usually:

1. evaluate all candidate contours
2. choose one best candidate for the frame
3. append only that center point to the trajectory

This is the transition from detection to tracking.

# Lesson 39 — A Choppy Trail Can Be Normal at Low FPS

A path can look spaced out when the object moves fast, especially if the video frame rate is low.

In this project, the video is about 23 FPS, so the ball may move a large distance between frames.

This does not necessarily mean points need to be added more often.

It may simply reflect the true sampling rate of the video.

# Lesson 40 — Storing the Best Candidate Is Not Enough if You Still Use the Last Loop Variable

A common bug in tracking code is:

1. correctly identify the best candidate
2. store it
3. accidentally append or draw using the last contour variables from the loop

This causes the tracker to jump to the wrong object.

Fix:

After the contour loop, unpack the saved `best_candidate` and use those values for drawing and path updates.

# Lesson 41 — “Best Candidate” Needs Temporal Consistency

Choosing the most circular contour in each frame is not enough when multiple ball-like objects are present.

Example:

- a moving basketball
- a stationary basketball on the floor

Both can have good circularity.

If the tracker chooses the best candidate independently each frame, it may switch between objects.

This means tracking requires temporal consistency.

A common next rule is:

- on the first frame, choose the best candidate by appearance
- on later frames, choose the candidate closest to the previously tracked center

# Lesson 42 — Nearest-Neighbor Tracking

Once a tracked object has a previous center point, the next candidate should usually be chosen based on distance to that previous point.

Example:

```python
last_center = ball_path[-1]
distance = math.hypot(center_x - last_center[0], center_y - last_center[1])
```

# Lesson 43 - Computing a Score Is Not the Same as Using it

A common tracking bug is:

1. compute a useful metric like distance to the last center
2. but still choose candidates using an older rule like max circularity

This means the tracker appears unchanged even though new logic was added.

To fix this, the candidate selection rule itself must switch:

- first frame: maximize circularity
- later frames: minimize distance to previous center
