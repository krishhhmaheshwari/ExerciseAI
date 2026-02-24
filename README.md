# HabitHealth SquatAI SDK

Real-time AI-powered squat analysis for browser and mobile. Counts reps, detects form errors, explains **what's wrong**, **why it happens**, and **how to fix it**.

## Quick Start

```javascript
import { SquatAISDK } from 'habithealth-squat-ai';

const squat = new SquatAISDK({
  camera: 'webcam',
  sessionDuration: 300,  // 5 minutes
  userId: 'employee_123',
  enableVoice: true,
});

// Get your video and canvas elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

// Start session (auto-calibrates first)
await squat.start(video, canvas);

// Listen for events
squat.onRep((rep) => {
  console.log(`Rep ${rep.repNumber}: ${rep.score}/100`);
});

squat.onFormError((err) => {
  console.log(`${err.type}: ${err.fix}`);
});

squat.onSessionComplete((report) => {
  console.log(`${report.totalReps} reps, mobility score: ${report.mobilityScore}`);
});

// Stop when done
const report = squat.stop();
squat.downloadReport(); // Save JSON report
```

## Architecture

```
/squat-ai-sdk
  ├── index.ts           → Main SDK class (SquatAISDK)
  ├── poseEngine.ts      → MediaPipe wrapper, angle math, skeleton drawing
  ├── squatAnalyzer.ts   → Rep counting FSM, form error detection
  ├── feedbackEngine.ts  → Human feedback text, voice cues
  ├── sessionManager.ts  → Session lifecycle, calibration, timer
  ├── api.ts             → Report export, storage, video training
  ├── types.ts           → All TypeScript interfaces
  └── README.md          → This file
```

## How It Works

### 3-Layer Pipeline (runs every frame at 30 FPS)

1. **Camera → MediaPipe Pose (BlazePose CNN)**  
   Webcam frame → 33 body landmarks (x, y, z, visibility)

2. **Landmarks → Angle Computation**  
   6 joint angles calculated via trigonometry + EMA smoothing

3. **Angles → Squat Analysis**  
   Finite state machine (standing → descending → bottom → ascending → standing = 1 rep)

### Form Detection (6 parallel checks)

| Error | Detection Method | Body Problem | Fix |
|-------|-----------------|--------------|-----|
| Forward Lean | Shoulder-hip angle vs vertical | Lower back strain | Keep chest up |
| Knee Collapse | Knee x vs ankle x position | ACL/meniscus risk | Push knees out |
| Knee Over Toe | Knee forward travel | Patellar stress | Sit back more |
| Shallow Squat | Knee angle at bottom | Reduced effectiveness | Elevate heels |
| Fast Tempo | Rep duration < 1s | Lost muscle activation | Count to 2 down |
| Asymmetry | |Left − Right knee| | Muscle imbalance | Single-leg work |

## Session Flow

1. **Calibration** (auto, ~1 second) — Records standing posture baseline
2. **Active Tracking** (2-10 minutes) — Counts reps, checks form each frame
3. **Report** — JSON with reps, errors, mobility score, recommendations

## Configuration

```javascript
new SquatAISDK({
  camera: 'webcam',           // 'webcam' or 'environment' (rear camera)
  sessionDuration: 120,       // seconds (default 120)
  userId: 'user_123',         // for report tracking
  enableVoice: true,          // spoken real-time cues
  enableOverlay: true,        // skeleton drawn on canvas
  thresholds: {               // override defaults
    kneeDown: 140,            // angle below = squatting
    kneeBottom: 100,          // angle below = full depth
    kneeUp: 160,              // angle above = standing
    backLeanMax: 35,          // max forward lean degrees
    smoothingAlpha: 0.25,     // EMA smoothing factor
  }
});
```

## Report Format

```json
{
  "totalReps": 18,
  "correctReps": 14,
  "errors": {
    "forwardLean": 3,
    "kneeCollapse": 2,
    "shallowSquats": 4,
    "fastTempo": 1,
    "kneeOverToe": 0,
    "heelLift": 0,
    "asymmetry": 1
  },
  "mobilityScore": 82,
  "riskLevel": "Medium",
  "averageDepth": 78,
  "averageRepTime": 2.1,
  "recommendations": [
    "Strengthen your core with planks to reduce forward lean.",
    "Add banded walks to strengthen glute medius."
  ],
  "disclaimer": "This tool provides fitness guidance only..."
}
```

## API Reference

### SquatAISDK

| Method | Description |
|--------|-------------|
| `start(video, canvas)` | Start session with auto-calibration |
| `stop()` | End session, returns report |
| `pause()` / `resume()` | Pause/resume tracking |
| `onRep(callback)` | Called when rep completes |
| `onFormError(callback)` | Called on form error detected |
| `onSessionComplete(callback)` | Called when session ends |
| `onFrame(callback)` | Called every processed frame |
| `getRepCount()` | Current rep count |
| `getState()` | 'idle' / 'calibrating' / 'active' / 'paused' / 'complete' |
| `getElapsed()` | Seconds elapsed |
| `getRemaining()` | Seconds remaining |
| `downloadReport()` | Download JSON report |
| `exportJSON()` | Get report as JSON string |
| `exportCSV()` | Get report as CSV string |
| `trainWithVideo(file)` | Extract training data from video |

### Static Methods

| Method | Description |
|--------|-------------|
| `SquatAISDK.getHistory()` | Get all saved session reports |
| `SquatAISDK.clearHistory()` | Clear saved sessions |

## Tech Stack

- **MediaPipe Pose** (BlazePose CNN) — 33-landmark detection, 30+ FPS
- **WebRTC** — Camera access
- **Canvas API** — Skeleton overlay
- **Web Speech API** — Voice feedback (optional)
- **100% client-side** — No server, no video upload, complete privacy

## Accuracy Notes

100% accuracy is scientifically unrealistic due to camera angles, lighting, occlusion, and body variability. This SDK achieves high accuracy through:

- Auto-calibration to user's standing baseline
- EMA smoothing to filter noise
- Hysteresis (hold frames) to prevent false triggers
- Tunable thresholds per-user
- 6 parallel form checks with severity scoring

## License

Proprietary — HabitHealth Ltd.
# MSK
