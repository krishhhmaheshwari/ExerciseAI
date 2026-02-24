#!/usr/bin/env python3
"""
Multi-Exercise JSON Analyzer — Reads COCO keypoint JSONs for all exercises.
No dependencies. Runs instantly.

USAGE: python3 exercise_json_analyzer.py
"""

import json, math, os, glob, sys

# ═══ COCO 17-keypoint indices ═══
NOSE=0
L_SHOULDER,R_SHOULDER=5,6
L_ELBOW,R_ELBOW=7,8
L_WRIST,R_WRIST=9,10
L_HIP,R_HIP=11,12
L_KNEE,R_KNEE=13,14
L_ANKLE,R_ANKLE=15,16

BASE=os.path.expanduser("~/Downloads/archive")
SYNTH=os.path.join(BASE,"synthetic_dataset/synthetic_dataset")

EXERCISES={
    'barbell_biceps_curl':{
        'folder':'barbell biceps curl',
        'needed':[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP],
    },
    'push_up':{
        'folder':'push-up',
        'needed':[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANKLE,R_ANKLE],
    },
    'shoulder_press':{
        'folder':'shoulder press',
        'needed':[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP],
    },
}

def parse_kp(flat):
    pts=[]
    for i in range(0,len(flat),3):
        pts.append((flat[i],flat[i+1],flat[i+2]))
    return pts

def angle_3pt(a,b,c):
    ba=[a[0]-b[0],a[1]-b[1]]
    bc=[c[0]-b[0],c[1]-b[1]]
    dot=ba[0]*bc[0]+ba[1]*bc[1]
    m1=math.sqrt(ba[0]**2+ba[1]**2)
    m2=math.sqrt(bc[0]**2+bc[1]**2)
    if m1*m2==0:return None
    cos_a=max(-1,min(1,dot/(m1*m2)))
    return math.degrees(math.acos(cos_a))

def angle_from_vertical(top,bottom):
    dx=bottom[0]-top[0]
    dy=bottom[1]-top[1]
    if dy==0:return 90
    return abs(math.degrees(math.atan(dx/dy)))

def pt(pts,i):return(pts[i][0],pts[i][1])

def visible(pts,indices):
    for i in indices:
        if i>=len(pts) or pts[i][2]==0:return False
    return True

# ═══ EXERCISE-SPECIFIC ANGLE COMPUTATION ═══

def compute_bicep_curl(pts):
    if not visible(pts,[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP]):
        return None
    # Elbow flexion (shoulder→elbow→wrist)
    el_l=angle_3pt(pt(pts,L_SHOULDER),pt(pts,L_ELBOW),pt(pts,L_WRIST))
    el_r=angle_3pt(pt(pts,R_SHOULDER),pt(pts,R_ELBOW),pt(pts,R_WRIST))
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    # Shoulder stability: shoulder angle (hip→shoulder→elbow) — should stay ~0 for strict curl
    sh_l=angle_3pt(pt(pts,L_HIP),pt(pts,L_SHOULDER),pt(pts,L_ELBOW))
    sh_r=angle_3pt(pt(pts,R_HIP),pt(pts,R_SHOULDER),pt(pts,R_ELBOW))
    shoulder=sh_l if sh_r is None else(sh_r if sh_l is None else(sh_l+sh_r)/2)
    if shoulder is None:shoulder=0

    # Torso swing (spine from vertical)
    mid_sh=((pts[L_SHOULDER][0]+pts[R_SHOULDER][0])/2,(pts[L_SHOULDER][1]+pts[R_SHOULDER][1])/2)
    mid_hp=((pts[L_HIP][0]+pts[R_HIP][0])/2,(pts[L_HIP][1]+pts[R_HIP][1])/2)
    torso=angle_from_vertical(mid_sh,mid_hp)

    # Wrist alignment relative to elbow (lateral deviation)
    def wrist_dev(e,w):
        dx=abs(w[0]-e[0])
        dy=abs(w[1]-e[1])
        if dy<0.5:return 45
        return math.degrees(math.atan(dx/dy))
    wd_l=wrist_dev(pt(pts,L_ELBOW),pt(pts,L_WRIST))
    wd_r=wrist_dev(pt(pts,R_ELBOW),pt(pts,R_WRIST))
    wrist_align=(wd_l+wd_r)/2

    return {
        'elbow':round(elbow,2),
        'shoulder_angle':round(shoulder,2),
        'torso_swing':round(torso,2),
        'wrist_align':round(wrist_align,2),
    }

def compute_pushup(pts):
    needed=[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP,L_ANKLE,R_ANKLE]
    if not visible(pts,needed):return None

    # Elbow flexion
    el_l=angle_3pt(pt(pts,L_SHOULDER),pt(pts,L_ELBOW),pt(pts,L_WRIST))
    el_r=angle_3pt(pt(pts,R_SHOULDER),pt(pts,R_ELBOW),pt(pts,R_WRIST))
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    # Body alignment (shoulder→hip→ankle — should be ~180° for plank)
    ba_l=angle_3pt(pt(pts,L_SHOULDER),pt(pts,L_HIP),pt(pts,L_ANKLE))
    ba_r=angle_3pt(pt(pts,R_SHOULDER),pt(pts,R_HIP),pt(pts,R_ANKLE))
    body_line=ba_l if ba_r is None else(ba_r if ba_l is None else(ba_l+ba_r)/2)
    if body_line is None:body_line=180

    # Hip sag/pike: deviation from 180° line
    hip_dev=abs(180-body_line)

    # Shoulder angle (elbow→shoulder→hip)
    sh_l=angle_3pt(pt(pts,L_ELBOW),pt(pts,L_SHOULDER),pt(pts,L_HIP))
    sh_r=angle_3pt(pt(pts,R_ELBOW),pt(pts,R_SHOULDER),pt(pts,R_HIP))
    shoulder=sh_l if sh_r is None else(sh_r if sh_l is None else(sh_l+sh_r)/2)
    if shoulder is None:shoulder=90

    return {
        'elbow':round(elbow,2),
        'body_line':round(body_line,2),
        'hip_deviation':round(hip_dev,2),
        'shoulder_angle':round(shoulder,2),
    }

def compute_shoulder_press(pts):
    if not visible(pts,[L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP]):
        return None

    # Elbow flexion
    el_l=angle_3pt(pt(pts,L_SHOULDER),pt(pts,L_ELBOW),pt(pts,L_WRIST))
    el_r=angle_3pt(pt(pts,R_SHOULDER),pt(pts,R_ELBOW),pt(pts,R_WRIST))
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    # Shoulder abduction (hip→shoulder→elbow)
    ab_l=angle_3pt(pt(pts,L_HIP),pt(pts,L_SHOULDER),pt(pts,L_ELBOW))
    ab_r=angle_3pt(pt(pts,R_HIP),pt(pts,R_SHOULDER),pt(pts,R_ELBOW))
    abduction=ab_l if ab_r is None else(ab_r if ab_l is None else(ab_l+ab_r)/2)
    if abduction is None:abduction=90

    # Torso lean
    mid_sh=((pts[L_SHOULDER][0]+pts[R_SHOULDER][0])/2,(pts[L_SHOULDER][1]+pts[R_SHOULDER][1])/2)
    mid_hp=((pts[L_HIP][0]+pts[R_HIP][0])/2,(pts[L_HIP][1]+pts[R_HIP][1])/2)
    torso=angle_from_vertical(mid_sh,mid_hp)

    # Lockout: wrist above shoulder (wrist.y < shoulder.y in image coords)
    lw_above=1 if pts[L_WRIST][1]<pts[L_SHOULDER][1] else 0
    rw_above=1 if pts[R_WRIST][1]<pts[R_SHOULDER][1] else 0
    lockout=(lw_above+rw_above)/2

    return {
        'elbow':round(elbow,2),
        'shoulder_abduction':round(abduction,2),
        'torso_lean':round(torso,2),
        'lockout':round(lockout,2),
    }

COMPUTE_FN={
    'barbell_biceps_curl':compute_bicep_curl,
    'push_up':compute_pushup,
    'shoulder_press':compute_shoulder_press,
}

# ═══ VALLEY DETECTION (exercise-specific) ═══
def detect_valleys(frames, primary_key, is_min_valley=True):
    """Detect rep bottoms. is_min_valley=True for exercises where primary metric decreases (curls, press)."""
    if not frames:return []
    vals=[f[primary_key] for f in frames]
    p90=percentile(vals,90)
    p10=percentile(vals,10)
    rng=p90-p10
    if rng<10:return []

    DESCENT=rng*0.3
    ASCENT=rng*0.2
    valleys=[]
    state='WAITING'
    if is_min_valley:
        start_thresh=p90-DESCENT
        deepest=999
        deepest_frame=None
        for i,f in enumerate(frames):
            v=f[primary_key]
            if state=='WAITING':
                if v<start_thresh:
                    state='DESCENDING';deepest=v;deepest_frame=f
            elif state=='DESCENDING':
                if v<deepest:deepest=v;deepest_frame=f
                if v>deepest+ASCENT:
                    valleys.append(deepest_frame)
                    state='WAITING';deepest=999
    else:
        # Max valley (for exercises where primary increases at bottom like push-up body_line)
        start_thresh=p10+DESCENT
        peak=0
        peak_frame=None
        for i,f in enumerate(frames):
            v=f[primary_key]
            if state=='WAITING':
                if v>start_thresh:
                    state='ASCENDING';peak=v;peak_frame=f
            elif state=='ASCENDING':
                if v>peak:peak=v;peak_frame=f
                if v<peak-ASCENT:
                    valleys.append(peak_frame)
                    state='WAITING';peak=0
    return valleys

def percentile(arr,p):
    if not arr:return 0
    s=sorted(arr)
    k=(len(s)-1)*p/100
    f=int(k);c=min(f+1,len(s)-1)
    d=k-f
    return s[f]*(1-d)+s[c]*d

def stats(values):
    if not values:return{'n':0}
    values=sorted(values)
    n=len(values);mean=sum(values)/n
    std=math.sqrt(sum((v-mean)**2 for v in values)/n)
    return{
        'n':n,'mean':round(mean,2),'std':round(std,2),
        'min':round(min(values),2),'max':round(max(values),2),
        'p10':round(percentile(values,10),2),
        'p25':round(percentile(values,25),2),
        'p50':round(percentile(values,50),2),
        'p75':round(percentile(values,75),2),
        'p90':round(percentile(values,90),2),
    }

# ═══ PRIMARY METRIC + VALLEY CONFIG ═══
EXERCISE_CONFIG={
    'barbell_biceps_curl':{'primary':'elbow','is_min':True,'metrics':['elbow','shoulder_angle','torso_swing','wrist_align']},
    'push_up':{'primary':'elbow','is_min':True,'metrics':['elbow','body_line','hip_deviation','shoulder_angle']},
    'shoulder_press':{'primary':'elbow','is_min':True,'metrics':['elbow','shoulder_abduction','torso_lean','lockout']},
}

def main():
    print("\n"+"="*60)
    print("  Multi-Exercise JSON Analyzer")
    print("="*60)

    all_results={}

    for ex_key,ex_info in EXERCISES.items():
        folder=os.path.join(SYNTH,ex_info['folder'])
        jsons=sorted(glob.glob(os.path.join(folder,"*.json")))
        if not jsons:
            print(f"\n  ⚠ No JSONs found for {ex_key}")
            continue

        print(f"\n  📂 {ex_info['folder']}: {len(jsons)} JSON files")
        compute_fn=COMPUTE_FN[ex_key]
        config=EXERCISE_CONFIG[ex_key]

        all_frames=[]
        all_valleys=[]
        all_resting=[]
        files_ok=0

        for jpath in jsons:
            with open(jpath) as f:
                try:data=json.load(f)
                except:continue
            if 'annotations' not in data:continue

            frames=[]
            for ann in data['annotations']:
                if 'keypoints' not in ann:continue
                pts=parse_kp(ann['keypoints'])
                angles=compute_fn(pts)
                if angles:
                    frames.append(angles)
                    all_frames.append(angles)

            if not frames:continue
            files_ok+=1

            # Detect valleys (rep bottoms)
            valleys=detect_valleys(frames,config['primary'],config['is_min'])
            all_valleys.extend(valleys)

            # Detect resting (top position)
            pvals=[f[config['primary']] for f in frames]
            if config['is_min']:
                p80=percentile(pvals,80)
                resting=[f for f in frames if f[config['primary']]>=p80]
            else:
                p20=percentile(pvals,20)
                resting=[f for f in frames if f[config['primary']]<=p20]
            all_resting.extend(resting)

            print(f"    ✓ {os.path.basename(jpath)}: {len(frames)} frames, {len(valleys)} reps")

        reps_found=len(all_valleys)
        print(f"\n  {ex_key}: {files_ok} files, {len(all_frames)} frames, {reps_found} reps detected")

        # Build stats
        result={
            '_meta':{'exercise':ex_key,'json_files':files_ok,'total_frames':len(all_frames),'reps_detected':reps_found,'resting_frames':len(all_resting)},
            'rep_bottom':{},
            'resting':{},
            'all_frames':{},
            'suggested_thresholds':{},
        }
        for m in config['metrics']:
            result['rep_bottom'][m]=stats([v[m] for v in all_valleys])
            result['resting'][m]=stats([r[m] for r in all_resting])
            result['all_frames'][m]=stats([f[m] for f in all_frames])

        # Suggested thresholds from valleys
        if all_valleys:
            for m in config['metrics']:
                vals=[v[m] for v in all_valleys]
                result['suggested_thresholds'][m]={
                    'ideal':round(percentile(vals,50),1),
                    'range_low':round(percentile(vals,10),1),
                    'range_high':round(percentile(vals,90),1),
                }

        all_results[ex_key]=result

        # Print summary table
        if all_valleys:
            print(f"\n  SUGGESTED THRESHOLDS ({ex_key}):")
            print(f"  ┌──────────────────────┬──────────┬─────────────────────┐")
            print(f"  │ Metric               │  Ideal   │  Range              │")
            print(f"  ├──────────────────────┼──────────┼─────────────────────┤")
            for m in config['metrics']:
                st=result['suggested_thresholds'][m]
                print(f"  │ {m:<20} │ {st['ideal']:>6.1f}°  │ {st['range_low']:>5.1f}° – {st['range_high']:>5.1f}°        │")
            print(f"  └──────────────────────┴──────────┴─────────────────────┘")

    # Save
    output_path=os.path.expanduser("~/Desktop/exercise_analysis_synthetic.json")
    with open(output_path,'w') as f:
        json.dump(all_results,f,indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Saved to: {output_path}")
    print(f"     Upload this file to Claude!")
    print(f"{'='*60}\n")

if __name__=='__main__':
    main()
