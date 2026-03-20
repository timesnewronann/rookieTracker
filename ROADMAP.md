# 🏀 ShotTracker Roadmap (v1 – Ground-Up Build)

## 📅 Project Start Date

**March 19, 2026**

---

# 🎯 Final Vision

A system where a user can:

- Upload or record a basketball shooting video
- Automatically track the ball
- Analyze shot arc and entry angle
- Receive actionable feedback

---

# 🧱 PHASE 1 — BALL DETECTION

## 📅 Timeline

**March 19 → March 26**

## 🎯 Goal

Detect the basketball reliably in each frame

## 🧠 Concepts

- HSV color space
- Color masking
- Contours
- Area filtering
- Aspect ratio
- Noise reduction

## 🔧 Tasks

- [x] Tune HSV range for basketball color
- [x] Visualize mask, contours, and debug overlays
- [x] Filter contours by area
- [x] Add **static ROI** to reduce search space
- [x] Improve mask quality (reduce fragmentation)

## ✅ Milestone

> “The system consistently detects the basketball in most frames of a controlled video”

## 📦 Deliverable

- Working script that highlights the basketball
- Debug modes: frame / mask / result

---

# 🧱 PHASE 2 — BALL TRACKING

## 📅 Timeline

**March 27 → April 2**

## 🎯 Goal

Track the SAME basketball across frames

## 🧠 Concepts

- Temporal consistency
- Nearest neighbor tracking
- Motion constraints
- Dynamic ROI

## 🔧 Tasks

- [ ] Store previous ball position
- [ ] Select closest contour each frame
- [ ] Implement **dynamic ROI**
- [ ] Handle lost tracking (fallback logic)

## ✅ Milestone

> “The system draws a continuous trajectory of the ball”

## 📦 Deliverable

- List of (x, y) ball positions over time
- Visual trajectory overlay

---

# 🧱 PHASE 3 — SHOT SEGMENTATION

## 📅 Timeline

**April 3 → April 9**

## 🎯 Goal

Detect when a shot starts and ends

## 🧠 Concepts

- Velocity (dy)
- Direction change
- State machines

## 🔧 Tasks

- [ ] Detect upward motion (shot start)
- [ ] Detect peak of trajectory
- [ ] Detect downward motion
- [ ] Segment individual shots

## ✅ Milestone

> “The system isolates a single shot from continuous footage”

## 📦 Deliverable

- Start and end frame indices for each shot

---

# 🧱 PHASE 4 — TRAJECTORY & ANALYSIS

## 📅 Timeline

**April 10 → April 20**

## 🎯 Goal

Extract meaningful shot metrics

## 🧠 Concepts

- Parabolic curve fitting
- Coordinate systems
- Basic physics

## 🔧 Tasks

- [ ] Fit trajectory curve (parabola)
- [ ] Calculate arc height
- [ ] Calculate release angle
- [ ] Calculate entry angle

## ✅ Milestone (🔥 RESUME LEVEL)

> “The system computes arc and entry angle from real shot data”

## 📦 Deliverable

- Numerical shot metrics
- Visual overlay of trajectory curve

---

# 🧱 PHASE 5 — MAKE / MISS DETECTION

## 📅 Timeline

**April 21 → April 27**

## 🎯 Goal

Determine shot outcome

## 🧠 Concepts

- Spatial reasoning
- Event detection

## 🔧 Tasks

- [ ] Define hoop region
- [ ] Detect ball passing through hoop area
- [ ] Classify make vs miss

## ✅ Milestone

> “The system classifies shot outcome (make/miss)”

## 📦 Deliverable

- Shot result label

---

# 🧱 PHASE 6 — CLI PIPELINE

## 📅 Timeline

**April 28 → May 5**

## 🎯 Goal

Turn project into a usable tool

## 🧠 Concepts

- CLI design
- Modular pipelines
- Input/output handling

## 🔧 Tasks

- [ ] Build `run_pipeline.py`
- [ ] Add argument parsing
- [ ] Output JSON results
- [ ] Generate overlay video

## ✅ Milestone (🚀 BIG)

> “Anyone can run the tool on their own video”

## 📦 Deliverable

### Command

```bash
python run_pipeline.py --video input.mp4 --out results.json --overlay
```

### Output

```json
{
  "arc": 48.2,
  "entry_angle": 42.1,
  "result": "make"
}
```

---

# 🧱 PHASE 7 — WEB APP

## 📅 Timeline

**May 6 → May 20**

## 🎯 Goal

Make the system accessible to users

## 🧠 Concepts

- Frontend/backend integration
- File uploads
- API design

## 🔧 Tasks

- [ ] Build simple UI (React / Next.js)
- [ ] Upload video feature
- [ ] Connect to Python backend
- [ ] Display results visually

## ✅ Milestone

> “Users can upload a video and see shot analysis”

---

# 🧱 PHASE 8 — LIVE TRACKING (ADVANCED)

## 📅 Timeline

**Future Phase (Post-MVP)**

## 🎯 Goal

Real-time shot tracking

## 🧠 Concepts

- Real-time CV
- Performance optimization
- Model inference (YOLO)

## 🔧 Tasks

- [ ] Replace HSV with object detection model
- [ ] Optimize pipeline speed
- [ ] Integrate webcam/live feed

## ✅ Milestone

> “System provides live shot feedback”

---

# 🧠 Key Principles

- Build **one phase at a time**
- Each phase must be:
  - Working
  - Testable
  - Demo-able

- Do NOT skip ahead

---

# 📍 Current Status

## 👉 You are here:

**PHASE 1 — Ball Detection**

---

# 🚀 Immediate Next Steps

- [ ] Add static ROI
- [ ] Improve mask quality
- [ ] Strengthen contour filtering

---

# 💼 Resume Goal

By Phase 4–6, this project should support a bullet like:

> Built a computer vision pipeline that processes basketball shooting videos to extract ball trajectory, arc, and entry angle, with modular CLI tooling and visual overlays for performance analysis.

---

# 🏁 End Goal

A production-ready system that evolves toward:

- Live tracking
- Mobile integration
- Athlete feedback platform

---
