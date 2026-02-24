#!/usr/bin/env python3
"""
SquatAI Dataset Processor v1.0
==============================
Run this script on your Mac to process squat video datasets.
It extracts biomechanical angles from all squat videos using MediaPipe,
identifies squat bottom positions, and outputs a JSON summary file
that can be uploaded to tune SquatAI v14.1 thresholds.

USAGE:
    pip install mediapipe opencv-python numpy
    python squatai_dataset_processor.py --input /path/to/video/folders --output squat_analysis.json

EXAMPLE:
    python squatai_dataset_processor.py \
        --input ~/Desktop/final_kaggle_with_additional_video ~/Desktop/similar_dataset \
        --output squat_analysis.json

The script will:
  1. Recursively find all video files (.mp4, .avi, .mov, .mkv)
  2. Filter for squat videos (by folder name or filename containing 'squat')
  3. Extract MediaPipe BlazePose landmarks from every frame
  4. Compute 5 golden standard angles (hip, knee, ankle, torso, valgus)
  5. Detect squat valleys (bottom positions) using a state machine
  6. Output statistical summary as a small JSON file (~1-2 KB)

Upload the JSON output to Claude, and it will integrate data-driven
thresholds into SquatAI v14.1.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
SQUAT_KEYWORDS = ['squat', 'sqt', 'bodyweight_squat', 'air_squat']
MIN_VISIBILITY = 0.4          # Minimum landmark visibility to trust
MIN_CORE_VISIBILITY = 0.35    # Minimum average core visibility per frame
KNEE_DESCENT_THRESHOLD = 20   # Degrees below standing to detect descent
KNEE_ASCENT_THRESHOLD = 12    # Degrees above bottom to detect ascent
KNEE_RETURN_THRESHOLD = 15    # Degrees from standing to count as returned
MIN_SQUAT_GAP_FRAMES = 15     # Minimum frames between squats (~0.5s at 30fps)

# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# ═══════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════

def angle_3pt(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))

def angle_from_vertical(top, bottom):
    """Angle of line top→bottom from vertical, in degrees."""
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    if dy == 0:
        return 90
    return abs(math.degrees(math.atan(dx / dy)))

def weighted_avg(val_l, vis_l, val_r, vis_r):
    """Visibility-weighted bilateral average."""
    if vis_l + vis_r < 0.01:
        return None
    return (val_l * vis_l + val_r * vis_r) / (vis_l + vis_r)

# ═══════════════════════════════════════════════════════════
# ANGLE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_angles(landmarks):
    """
    Extract 5 golden standard angles from MediaPipe landmarks.
    Returns dict with angles or None if visibility too low.
    """
    lm = landmarks

    # Core visibility check
    core_indices = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    core_vis = np.mean([lm[i].visibility for i in core_indices])
    if core_vis < MIN_CORE_VISIBILITY:
        return None

    # Extract coordinates
    def pt(idx):
        return (lm[idx].x, lm[idx].y)
    def vis(idx):
        return lm[idx].visibility

    # 1. Hip Flexion (shoulder → hip → knee)
    hip_l = angle_3pt(pt(L_SHOULDER), pt(L_HIP), pt(L_KNEE))
    hip_r = angle_3pt(pt(R_SHOULDER), pt(R_HIP), pt(R_KNEE))
    hip = weighted_avg(hip_l, vis(L_HIP), hip_r, vis(R_HIP))

    # 2. Knee Flexion (hip → knee → ankle)
    knee_l = angle_3pt(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))
    knee_r = angle_3pt(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))
    knee = weighted_avg(knee_l, vis(L_KNEE), knee_r, vis(R_KNEE))

    # 3. Ankle Dorsiflexion (shin-to-vertical)
    ankle_l = angle_from_vertical(pt(L_KNEE), pt(L_ANKLE))
    ankle_r = angle_from_vertical(pt(R_KNEE), pt(R_ANKLE))
    ankle = weighted_avg(ankle_l, vis(L_ANKLE), ankle_r, vis(R_ANKLE))

    # 4. Torso Lean (spine midline from vertical)
    mid_shoulder = ((lm[L_SHOULDER].x + lm[R_SHOULDER].x)/2,
                    (lm[L_SHOULDER].y + lm[R_SHOULDER].y)/2)
    mid_hip = ((lm[L_HIP].x + lm[R_HIP].x)/2,
               (lm[L_HIP].y + lm[R_HIP].y)/2)
    torso = angle_from_vertical(mid_shoulder, mid_hip)

    # 5. Knee Valgus (medial deviation)
    # Approximate: angle of knee from hip-ankle line in frontal plane
    def valgus_side(hip_idx, knee_idx, ankle_idx):
        h, k, a = pt(hip_idx), pt(knee_idx), pt(ankle_idx)
        # Expected knee X position (linear interpolation between hip and ankle)
        if abs(h[1] - a[1]) < 0.001:
            return 0
        t = (k[1] - h[1]) / (a[1] - h[1])
        expected_x = h[0] + t * (a[0] - h[0])
        deviation = k[0] - expected_x
        # Convert to approximate degrees (scaled by limb length)
        limb_len = math.sqrt((h[0]-a[0])**2 + (h[1]-a[1])**2)
        if limb_len < 0.001:
            return 0
        return math.degrees(math.atan(deviation / (limb_len * 0.5)))

    valgus_l = valgus_side(L_HIP, L_KNEE, L_ANKLE)
    valgus_r = valgus_side(R_HIP, R_KNEE, R_ANKLE)
    valgus = weighted_avg(abs(valgus_l), vis(L_KNEE), abs(valgus_r), vis(R_KNEE))

    if any(v is None for v in [hip, knee, ankle, torso, valgus]):
        return None

    return {
        'hip': round(hip, 2),
        'knee': round(knee, 2),
        'ankle': round(ankle, 2),
        'torso': round(torso, 2),
        'valgus': round(valgus, 2),
        'core_vis': round(core_vis, 3),
        'hip_y': round((lm[L_HIP].y + lm[R_HIP].y) / 2, 4)
    }

# ═══════════════════════════════════════════════════════════
# SQUAT VALLEY DETECTOR (State Machine)
# ═══════════════════════════════════════════════════════════

class SquatDetector:
    """
    State machine squat detector — same logic as SquatAI v14.1.
    Identifies squat bottom positions from angle sequences.
    """
    WAITING = 'WAITING'
    DESCENDING = 'DESCENDING'
    AT_BOTTOM = 'AT_BOTTOM'

    def __init__(self, standing_knee):
        self.state = self.WAITING
        self.standing_knee = standing_knee
        self.deepest_knee = 999
        self.deepest_frame = None
        self.last_squat_frame = -999
        self.valleys = []  # List of frame data at squat bottoms

    def update(self, frame_idx, angles):
        knee = angles['knee']

        if self.state == self.WAITING:
            if knee < self.standing_knee - KNEE_DESCENT_THRESHOLD:
                self.state = self.DESCENDING
                self.deepest_knee = knee
                self.deepest_frame = angles.copy()
                self.deepest_frame['frame'] = frame_idx

        elif self.state == self.DESCENDING:
            if knee < self.deepest_knee:
                self.deepest_knee = knee
                self.deepest_frame = angles.copy()
                self.deepest_frame['frame'] = frame_idx
            if knee > self.deepest_knee + KNEE_ASCENT_THRESHOLD:
                self.state = self.AT_BOTTOM

        elif self.state == self.AT_BOTTOM:
            if knee >= self.standing_knee - KNEE_RETURN_THRESHOLD:
                if frame_idx - self.last_squat_frame >= MIN_SQUAT_GAP_FRAMES:
                    # Valid squat completed
                    self.valleys.append(self.deepest_frame)
                    self.last_squat_frame = frame_idx
                self.state = self.WAITING
                self.deepest_knee = 999

# ═══════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════

def find_squat_videos(input_paths):
    """Recursively find squat video files."""
    videos = []
    for path in input_paths:
        p = Path(path)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(str(p))
        elif p.is_dir():
            for f in p.rglob('*'):
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    # Check if path or filename contains squat keywords
                    full_lower = str(f).lower()
                    if any(kw in full_lower for kw in SQUAT_KEYWORDS):
                        videos.append(str(f))
    return videos

def process_video(video_path, pose):
    """Process a single video and extract angle data for every frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_angles = []
    standing_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            angles = extract_angles(results.pose_landmarks.landmark)
            if angles:
                angles['frame'] = frame_idx
                all_angles.append(angles)

                # First 30 frames with high knee angle = likely standing
                if frame_idx < 30 and angles['knee'] > 150:
                    standing_frames.append(angles)

        frame_idx += 1

    cap.release()

    # Estimate standing knee angle
    if standing_frames:
        standing_knee = np.mean([f['knee'] for f in standing_frames])
    elif all_angles:
        # Fallback: use 90th percentile (most extended position)
        standing_knee = np.percentile([f['knee'] for f in all_angles], 90)
    else:
        return None, None, None

    return all_angles, standing_knee, fps

def process_all_videos(video_paths):
    """Process all videos and collect squat valley data."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    all_valleys = []        # Angle snapshots at squat bottoms
    all_standing = []       # Standing angle snapshots
    all_frame_angles = []   # Every valid frame's angles
    video_stats = []        # Per-video summary

    total = len(video_paths)
    print(f"\n{'='*60}")
    print(f"  SquatAI Dataset Processor v1.0")
    print(f"  Processing {total} squat video(s)")
    print(f"{'='*60}\n")

    for i, vpath in enumerate(video_paths):
        fname = os.path.basename(vpath)
        print(f"  [{i+1}/{total}] {fname}...", end='', flush=True)
        t0 = time.time()

        all_angles, standing_knee, fps = process_video(vpath, pose)
        if all_angles is None or len(all_angles) < 10:
            print(f" SKIPPED (too few valid frames)")
            continue

        # Detect squats
        detector = SquatDetector(standing_knee)
        for angles in all_angles:
            detector.update(angles['frame'], angles)

        n_squats = len(detector.valleys)
        dt = time.time() - t0
        print(f" {len(all_angles)} frames, {n_squats} squats detected ({dt:.1f}s)")

        # Collect standing frames (top 20% knee angles = most upright)
        knee_threshold = np.percentile([f['knee'] for f in all_angles], 80)
        standing = [f for f in all_angles if f['knee'] >= knee_threshold]

        all_valleys.extend(detector.valleys)
        all_standing.extend(standing)
        all_frame_angles.extend(all_angles)

        video_stats.append({
            'file': fname,
            'frames': len(all_angles),
            'squats': n_squats,
            'standing_knee': round(standing_knee, 1),
            'fps': round(fps, 1)
        })

    pose.close()
    return all_valleys, all_standing, all_frame_angles, video_stats

# ═══════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════

def compute_statistics(all_valleys, all_standing, all_frame_angles, video_stats):
    """Compute comprehensive statistics for SquatAI threshold tuning."""

    def stats_for(values, name):
        if not values:
            return {'n': 0, 'note': f'No data for {name}'}
        arr = np.array(values)
        return {
            'n': len(arr),
            'mean': round(float(np.mean(arr)), 2),
            'std': round(float(np.std(arr)), 2),
            'min': round(float(np.min(arr)), 2),
            'max': round(float(np.max(arr)), 2),
            'p5': round(float(np.percentile(arr, 5)), 2),
            'p10': round(float(np.percentile(arr, 10)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p50': round(float(np.percentile(arr, 50)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p90': round(float(np.percentile(arr, 90)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
        }

    result = {
        '_meta': {
            'generator': 'SquatAI Dataset Processor v1.0',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_videos': len(video_stats),
            'total_frames_analyzed': len(all_frame_angles),
            'total_squats_detected': len(all_valleys),
            'total_standing_frames': len(all_standing),
        },
        'squat_bottom': {
            'description': 'Angles at the deepest point of each detected squat',
            'hip': stats_for([v['hip'] for v in all_valleys], 'hip@bottom'),
            'knee': stats_for([v['knee'] for v in all_valleys], 'knee@bottom'),
            'ankle': stats_for([v['ankle'] for v in all_valleys], 'ankle@bottom'),
            'torso': stats_for([v['torso'] for v in all_valleys], 'torso@bottom'),
            'valgus': stats_for([v['valgus'] for v in all_valleys], 'valgus@bottom'),
        },
        'standing': {
            'description': 'Angles when standing upright (top 20% knee extension)',
            'hip': stats_for([s['hip'] for s in all_standing], 'hip@standing'),
            'knee': stats_for([s['knee'] for s in all_standing], 'knee@standing'),
            'ankle': stats_for([s['ankle'] for s in all_standing], 'ankle@standing'),
            'torso': stats_for([s['torso'] for s in all_standing], 'torso@standing'),
            'valgus': stats_for([s['valgus'] for s in all_standing], 'valgus@standing'),
        },
        'all_frames': {
            'description': 'Angles across ALL frames (entire movement range)',
            'hip': stats_for([f['hip'] for f in all_frame_angles], 'hip@all'),
            'knee': stats_for([f['knee'] for f in all_frame_angles], 'knee@all'),
            'ankle': stats_for([f['ankle'] for f in all_frame_angles], 'ankle@all'),
            'torso': stats_for([f['torso'] for f in all_frame_angles], 'torso@all'),
            'valgus': stats_for([f['valgus'] for f in all_frame_angles], 'valgus@all'),
        },
        'suggested_thresholds': {},
        'video_details': video_stats
    }

    # Compute suggested thresholds based on data
    if all_valleys:
        v_knee = [v['knee'] for v in all_valleys]
        v_hip = [v['hip'] for v in all_valleys]
        v_ankle = [v['ankle'] for v in all_valleys]
        v_torso = [v['torso'] for v in all_valleys]
        v_valgus = [v['valgus'] for v in all_valleys]

        result['suggested_thresholds'] = {
            'description': 'Data-driven thresholds computed from squat bottom positions',
            'hip_flexion': {
                'ideal': round(float(np.median(v_hip)), 1),
                'range_low': round(float(np.percentile(v_hip, 10)), 1),
                'range_high': round(float(np.percentile(v_hip, 90)), 1),
            },
            'knee_flexion': {
                'ideal': round(float(np.median(v_knee)), 1),
                'range_low': round(float(np.percentile(v_knee, 10)), 1),
                'range_high': round(float(np.percentile(v_knee, 90)), 1),
            },
            'ankle_dorsiflexion': {
                'ideal': round(float(np.median(v_ankle)), 1),
                'range_low': round(float(np.percentile(v_ankle, 10)), 1),
                'range_high': round(float(np.percentile(v_ankle, 90)), 1),
            },
            'torso_lean': {
                'ideal': round(float(np.median(v_torso)), 1),
                'range_low': round(float(np.percentile(v_torso, 10)), 1),
                'range_high': round(float(np.percentile(v_torso, 90)), 1),
            },
            'knee_valgus': {
                'ideal': 0.0,
                'warning_threshold': round(float(np.percentile(v_valgus, 75)), 1),
                'flag_threshold': round(float(np.percentile(v_valgus, 90)), 1),
            },
            'descent_trigger_offset': round(float(np.std(v_knee) * 0.5), 1),
        }

    return result

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SquatAI Dataset Processor — Extract biomechanical angle statistics from squat videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process a single folder:
  python squatai_dataset_processor.py --input ./final_kaggle_with_additional_video

  # Process multiple folders:
  python squatai_dataset_processor.py --input ./final_kaggle ./similar_dataset ./synthetic_dataset

  # Process all videos (not just squat-filtered):
  python squatai_dataset_processor.py --input ./my_videos --all-videos

  # Custom output path:
  python squatai_dataset_processor.py --input ./data --output ~/Desktop/squat_analysis.json
        """
    )
    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='Input folder(s) or video file(s) to process')
    parser.add_argument('--output', '-o', default='squat_analysis.json',
                        help='Output JSON file path (default: squat_analysis.json)')
    parser.add_argument('--all-videos', action='store_true',
                        help='Process ALL videos, not just ones with "squat" in path/name')

    args = parser.parse_args()

    # Find videos
    if args.all_videos:
        videos = []
        for path in args.input:
            p = Path(path)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(str(p))
            elif p.is_dir():
                for f in p.rglob('*'):
                    if f.suffix.lower() in VIDEO_EXTENSIONS:
                        videos.append(str(f))
    else:
        videos = find_squat_videos(args.input)

    if not videos:
        print(f"\nNo squat videos found in: {args.input}")
        print("Tips:")
        print("  - Make sure folder/filenames contain 'squat'")
        print("  - Use --all-videos flag to process all video files")
        print(f"  - Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)

    print(f"\nFound {len(videos)} squat video(s):")
    for v in videos[:10]:
        print(f"  • {os.path.basename(v)}")
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")

    # Process
    all_valleys, all_standing, all_frames, video_stats = process_all_videos(videos)

    if not all_valleys:
        print("\n⚠ No squats detected in any video!")
        print("  This could mean:")
        print("  - Videos don't contain squats")
        print("  - Camera angle makes pose detection unreliable")
        print("  - Videos are too short or low resolution")

    # Compute stats
    result = compute_statistics(all_valleys, all_standing, all_frames, video_stats)

    # Save
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Videos processed:    {result['_meta']['total_videos']}")
    print(f"  Total frames:        {result['_meta']['total_frames_analyzed']}")
    print(f"  Squats detected:     {result['_meta']['total_squats_detected']}")
    print(f"  Standing frames:     {result['_meta']['total_standing_frames']}")

    if 'knee_flexion' in result.get('suggested_thresholds', {}):
        st = result['suggested_thresholds']
        print(f"\n  SUGGESTED THRESHOLDS (from data):")
        print(f"  ┌─────────────────────┬──────────┬─────────────────────┐")
        print(f"  │ Metric              │  Ideal   │  Range              │")
        print(f"  ├─────────────────────┼──────────┼─────────────────────┤")
        print(f"  │ Hip Flexion         │ {st['hip_flexion']['ideal']:>6.1f}°  │ {st['hip_flexion']['range_low']:>5.1f}° – {st['hip_flexion']['range_high']:>5.1f}°        │")
        print(f"  │ Knee Flexion        │ {st['knee_flexion']['ideal']:>6.1f}°  │ {st['knee_flexion']['range_low']:>5.1f}° – {st['knee_flexion']['range_high']:>5.1f}°        │")
        print(f"  │ Ankle Dorsiflexion  │ {st['ankle_dorsiflexion']['ideal']:>6.1f}°  │ {st['ankle_dorsiflexion']['range_low']:>5.1f}° – {st['ankle_dorsiflexion']['range_high']:>5.1f}°        │")
        print(f"  │ Torso Lean          │ {st['torso_lean']['ideal']:>6.1f}°  │ {st['torso_lean']['range_low']:>5.1f}° – {st['torso_lean']['range_high']:>5.1f}°        │")
        print(f"  │ Knee Valgus (flag)  │   0.0°   │ warn>{st['knee_valgus']['warning_threshold']:.1f}° flag>{st['knee_valgus']['flag_threshold']:.1f}°  │")
        print(f"  └─────────────────────┴──────────┴─────────────────────┘")

    print(f"\n  Output saved to: {output_path.resolve()}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print(f"\n  → Upload this JSON file to Claude to integrate")
    print(f"    data-driven thresholds into SquatAI v14.1")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
