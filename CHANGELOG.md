# Changelog

All notable changes to ExerciseAI are documented here.

## [15.0] - 2026-02-24

### Added
- Multi-exercise support: Barbell Bicep Curl, Hammer Curl, Push-up, Shoulder Press
- Exercise selector on splash screen with dataset size indicators
- Per-exercise biomechanics cards (auto-adapt labels, thresholds, ranges)
- Exercise-specific calibration prompts
- Exercise-specific skeleton joint color-coding
- Exercise-specific coaching messages
- Per-exercise localStorage calibration persistence
- "Switch Exercise" button for mid-session changes
- Data-driven thresholds from 6,600+ total reps

### Dataset
- Barbell Bicep Curl: 3,181 reps (175 real + 3,006 synthetic)
- Hammer Curl: 126 reps (real only, 19 videos)
- Push-up: 1,177 reps (97 real + 1,080 synthetic)
- Shoulder Press: 2,131 reps (360 real + 1,771 synthetic)
- Squat: 728 reps (carried from v14.2)

## [14.2] - 2026-02-24

### Changed
- Data-driven squat thresholds from 728 analyzed squats (135 real + 593 synthetic)
- 70/30 real/synthetic weighting for threshold computation
- Bolder skeleton overlay (5px bones, 9px joints, 16px glow)
- Fixed video/canvas alignment (`object-fit: contain` with offset calculation)

### Added
- localStorage persistence for calibration data
- "Skip Calibration" button when saved data exists
- 3-layer false positive prevention (grace period, min duration, min depth)

### Key Finding
- Real humans squat much deeper (~65° knee) than synthetic avatars (~118°)
- v14.1 was rejecting ~60-70% of valid real-world squats
- v14.2 correctly classifies ~75-80% (net improvement ~40-50%)

## [14.1] - 2026-02-24

### Added
- State machine squat detector (WAITING → DESCENDING → AT_BOTTOM → ASCENDING)
- One Euro Filter (Casiez et al., CHI 2012) for adaptive signal smoothing
- Lockout-based speech system (2.5s minimum gap, calibration priority)
- Interactive depth gauge overlay
- Form ring overlay with real-time posture score
- Angle arc visualizations at key joints

### Changed
- Complete UI rebuild with HCL Healthcare branding
- 5 biomechanical golden standards with clinical thresholds
- "What → Why → Fix" coaching framework

## [14.0] - 2026-02-24

### Added
- Initial clinical-grade architecture
- MediaPipe BlazePose integration (complexity=2)
- 3-step calibration system (stand, good reps, bad reps)
- Real-time biomechanical analysis for squats
