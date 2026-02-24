#!/usr/bin/env python3
"""
Multi-Exercise Video Processor — MediaPipe BlazePose on real videos.
Processes: barbell biceps curl, hammer curl, push-up, shoulder press

USAGE:
    source ~/squat_env/bin/activate
    python3 exercise_video_processor.py
"""

import json, math, os, glob, sys, time

try:
    import cv2
    import mediapipe as mp
except ImportError:
    print("Install dependencies: pip install mediapipe==0.10.14 opencv-python numpy")
    sys.exit(1)

# ═══ MediaPipe BlazePose landmark indices ═══
L_SHOULDER,R_SHOULDER=11,12
L_ELBOW,R_ELBOW=13,14
L_WRIST,R_WRIST=15,16
L_HIP,R_HIP=23,24
L_KNEE,R_KNEE=25,26
L_ANKLE,R_ANKLE=27,28

BASE=os.path.expanduser("~/Downloads/archive")

# All exercise folders across all dataset directories
EXERCISES={
    'barbell_biceps_curl':{
        'folders':[
            os.path.join(BASE,"final_kaggle_with_additional_video/barbell biceps curl"),
            os.path.join(BASE,"similar_dataset/barbell biceps curl"),
            os.path.join(BASE,"my_test_video_1/barbell biceps curl"),
        ]
    },
    'hammer_curl':{
        'folders':[
            os.path.join(BASE,"final_kaggle_with_additional_video/hammer curl"),
        ]
    },
    'push_up':{
        'folders':[
            os.path.join(BASE,"final_kaggle_with_additional_video/push-up"),
            os.path.join(BASE,"similar_dataset/push-up"),
            os.path.join(BASE,"my_test_video_1/push-up"),
        ]
    },
    'shoulder_press':{
        'folders':[
            os.path.join(BASE,"final_kaggle_with_additional_video/shoulder press"),
            os.path.join(BASE,"similar_dataset/shoulder press"),
            os.path.join(BASE,"my_test_video_1/shoulder press"),
        ]
    },
}

def angle_3pt(a,b,c):
    ba=[a.x-b.x,a.y-b.y]
    bc=[c.x-b.x,c.y-b.y]
    dot=ba[0]*bc[0]+ba[1]*bc[1]
    m1=math.sqrt(ba[0]**2+ba[1]**2)
    m2=math.sqrt(bc[0]**2+bc[1]**2)
    if m1*m2==0:return None
    cos_a=max(-1,min(1,dot/(m1*m2)))
    return math.degrees(math.acos(cos_a))

def angle_from_vertical(top,bottom):
    dx=bottom.x-top.x
    dy=bottom.y-top.y
    if abs(dy)<0.001:return 90
    return abs(math.degrees(math.atan(dx/dy)))

def vis_ok(lm,*ids):
    return all(lm[i].visibility>0.4 for i in ids)

# ═══ EXERCISE-SPECIFIC COMPUTATIONS ═══

def compute_bicep_curl(lm):
    if not vis_ok(lm,L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP):
        return None
    el_l=angle_3pt(lm[L_SHOULDER],lm[L_ELBOW],lm[L_WRIST])
    el_r=angle_3pt(lm[R_SHOULDER],lm[R_ELBOW],lm[R_WRIST])
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    sh_l=angle_3pt(lm[L_HIP],lm[L_SHOULDER],lm[L_ELBOW])
    sh_r=angle_3pt(lm[R_HIP],lm[R_SHOULDER],lm[R_ELBOW])
    shoulder=sh_l if sh_r is None else(sh_r if sh_l is None else(sh_l+sh_r)/2)
    if shoulder is None:shoulder=0

    mid_sh_x=(lm[L_SHOULDER].x+lm[R_SHOULDER].x)/2
    mid_sh_y=(lm[L_SHOULDER].y+lm[R_SHOULDER].y)/2
    mid_hp_x=(lm[L_HIP].x+lm[R_HIP].x)/2
    mid_hp_y=(lm[L_HIP].y+lm[R_HIP].y)/2
    dx=abs(mid_sh_x-mid_hp_x)
    dy=abs(mid_hp_y-mid_sh_y)
    torso=abs(math.degrees(math.atan(dx/dy))) if dy>0.001 else 90

    def wrist_dev(e,w):
        dx2=abs(w.x-e.x);dy2=abs(w.y-e.y)
        if dy2<0.001:return 45
        return math.degrees(math.atan(dx2/dy2))
    wd=(wrist_dev(lm[L_ELBOW],lm[L_WRIST])+wrist_dev(lm[R_ELBOW],lm[R_WRIST]))/2

    return{'elbow':round(elbow,2),'shoulder_angle':round(shoulder,2),'torso_swing':round(torso,2),'wrist_align':round(wd,2)}

def compute_hammer_curl(lm):
    # Same metrics as bicep curl
    return compute_bicep_curl(lm)

def compute_pushup(lm):
    if not vis_ok(lm,L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP,L_ANKLE,R_ANKLE):
        return None
    el_l=angle_3pt(lm[L_SHOULDER],lm[L_ELBOW],lm[L_WRIST])
    el_r=angle_3pt(lm[R_SHOULDER],lm[R_ELBOW],lm[R_WRIST])
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    ba_l=angle_3pt(lm[L_SHOULDER],lm[L_HIP],lm[L_ANKLE])
    ba_r=angle_3pt(lm[R_SHOULDER],lm[R_HIP],lm[R_ANKLE])
    body_line=ba_l if ba_r is None else(ba_r if ba_l is None else(ba_l+ba_r)/2)
    if body_line is None:body_line=180
    hip_dev=abs(180-body_line)

    sh_l=angle_3pt(lm[L_ELBOW],lm[L_SHOULDER],lm[L_HIP])
    sh_r=angle_3pt(lm[R_ELBOW],lm[R_SHOULDER],lm[R_HIP])
    shoulder=sh_l if sh_r is None else(sh_r if sh_l is None else(sh_l+sh_r)/2)
    if shoulder is None:shoulder=90

    return{'elbow':round(elbow,2),'body_line':round(body_line,2),'hip_deviation':round(hip_dev,2),'shoulder_angle':round(shoulder,2)}

def compute_shoulder_press(lm):
    if not vis_ok(lm,L_SHOULDER,R_SHOULDER,L_ELBOW,R_ELBOW,L_WRIST,R_WRIST,L_HIP,R_HIP):
        return None
    el_l=angle_3pt(lm[L_SHOULDER],lm[L_ELBOW],lm[L_WRIST])
    el_r=angle_3pt(lm[R_SHOULDER],lm[R_ELBOW],lm[R_WRIST])
    if el_l is None and el_r is None:return None
    elbow=el_l if el_r is None else(el_r if el_l is None else(el_l+el_r)/2)

    ab_l=angle_3pt(lm[L_HIP],lm[L_SHOULDER],lm[L_ELBOW])
    ab_r=angle_3pt(lm[R_HIP],lm[R_SHOULDER],lm[R_ELBOW])
    abduction=ab_l if ab_r is None else(ab_r if ab_l is None else(ab_l+ab_r)/2)
    if abduction is None:abduction=90

    mid_sh_x=(lm[L_SHOULDER].x+lm[R_SHOULDER].x)/2
    mid_sh_y=(lm[L_SHOULDER].y+lm[R_SHOULDER].y)/2
    mid_hp_x=(lm[L_HIP].x+lm[R_HIP].x)/2
    mid_hp_y=(lm[L_HIP].y+lm[R_HIP].y)/2
    dx=abs(mid_sh_x-mid_hp_x);dy=abs(mid_hp_y-mid_sh_y)
    torso=abs(math.degrees(math.atan(dx/dy))) if dy>0.001 else 90

    lw_above=1 if lm[L_WRIST].y<lm[L_SHOULDER].y else 0
    rw_above=1 if lm[R_WRIST].y<lm[R_SHOULDER].y else 0
    lockout=(lw_above+rw_above)/2

    return{'elbow':round(elbow,2),'shoulder_abduction':round(abduction,2),'torso_lean':round(torso,2),'lockout':round(lockout,2)}

COMPUTE_FN={
    'barbell_biceps_curl':compute_bicep_curl,
    'hammer_curl':compute_hammer_curl,
    'push_up':compute_pushup,
    'shoulder_press':compute_shoulder_press,
}

EXERCISE_CONFIG={
    'barbell_biceps_curl':{'primary':'elbow','is_min':True,'metrics':['elbow','shoulder_angle','torso_swing','wrist_align']},
    'hammer_curl':{'primary':'elbow','is_min':True,'metrics':['elbow','shoulder_angle','torso_swing','wrist_align']},
    'push_up':{'primary':'elbow','is_min':True,'metrics':['elbow','body_line','hip_deviation','shoulder_angle']},
    'shoulder_press':{'primary':'elbow','is_min':True,'metrics':['elbow','shoulder_abduction','torso_lean','lockout']},
}

def percentile(arr,p):
    if not arr:return 0
    s=sorted(arr);k=(len(s)-1)*p/100
    f=int(k);c=min(f+1,len(s)-1);d=k-f
    return s[f]*(1-d)+s[c]*d

def stats(values):
    if not values:return{'n':0}
    values=sorted(values);n=len(values);mean=sum(values)/n
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

def detect_valleys(frames,primary_key,is_min=True):
    if not frames:return[]
    vals=[f[primary_key] for f in frames]
    p90=percentile(vals,90);p10=percentile(vals,10)
    rng=p90-p10
    if rng<10:return[]
    DESCENT=rng*0.3;ASCENT=rng*0.2
    valleys=[];state='WAITING'
    if is_min:
        thresh=p90-DESCENT;deepest=999;df=None
        for f in frames:
            v=f[primary_key]
            if state=='WAITING':
                if v<thresh:state='DESC';deepest=v;df=f
            elif state=='DESC':
                if v<deepest:deepest=v;df=f
                if v>deepest+ASCENT:valleys.append(df);state='WAITING';deepest=999
    return valleys

def process_videos(ex_key,folders,compute_fn,config):
    mp_pose=mp.solutions.pose
    pose=mp_pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

    all_frames=[];all_valleys=[];all_resting=[];video_details=[]
    total_vids=0

    for folder in folders:
        if not os.path.isdir(folder):continue
        vids=[]
        for ext in['*.mp4','*.MOV','*.mov','*.avi','*.AVI']:
            vids.extend(glob.glob(os.path.join(folder,ext)))
        vids.sort()
        if not vids:continue
        total_vids+=len(vids)
        print(f"\n  📂 {os.path.relpath(folder,BASE)}: {len(vids)} videos")

        for vi,vpath in enumerate(vids):
            fname=os.path.basename(vpath)
            cap=cv2.VideoCapture(vpath)
            if not cap.isOpened():print(f"    ✗ {fname}: can't open");continue
            fps_v=cap.get(cv2.CAP_PROP_FPS) or 25
            frames=[]
            fc=0
            while True:
                ret,frame=cap.read()
                if not ret:break
                fc+=1
                if fc%2!=0:continue  # Process every 2nd frame for speed
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=pose.process(rgb)
                if not res.pose_landmarks:continue
                lm=res.pose_landmarks.landmark
                angles=compute_fn(lm)
                if angles:
                    frames.append(angles)
                    all_frames.append(angles)
            cap.release()
            if not frames:print(f"    ✗ {fname}: no landmarks");continue

            valleys=detect_valleys(frames,config['primary'],config['is_min'])
            all_valleys.extend(valleys)

            pvals=[f[config['primary']] for f in frames]
            p80=percentile(pvals,80)
            resting=[f for f in frames if f[config['primary']]>=p80]
            all_resting.extend(resting)

            video_details.append({'file':fname,'frames':len(frames),'reps':len(valleys),'fps':round(fps_v,1)})
            print(f"    ✓ {fname}: {len(frames)} frames, {len(valleys)} reps")

    pose.close()
    return all_frames,all_valleys,all_resting,video_details,total_vids

def main():
    print("\n"+"="*60)
    print("  Multi-Exercise Video Processor (MediaPipe)")
    print("="*60)

    all_results={}
    t0=time.time()

    for ex_key,ex_info in EXERCISES.items():
        config=EXERCISE_CONFIG[ex_key]
        compute_fn=COMPUTE_FN[ex_key]
        print(f"\n{'─'*60}")
        print(f"  Processing: {ex_key}")
        print(f"{'─'*60}")

        all_frames,all_valleys,all_resting,video_details,total_vids=process_videos(
            ex_key,ex_info['folders'],compute_fn,config)

        print(f"\n  {ex_key}: {total_vids} videos, {len(all_frames)} frames, {len(all_valleys)} reps")

        result={
            '_meta':{'exercise':ex_key,'total_videos':total_vids,'total_frames':len(all_frames),
                     'reps_detected':len(all_valleys),'resting_frames':len(all_resting)},
            'rep_bottom':{},
            'resting':{},
            'all_frames':{},
            'suggested_thresholds':{},
            'video_details':video_details,
        }
        for m in config['metrics']:
            result['rep_bottom'][m]=stats([v[m] for v in all_valleys])
            result['resting'][m]=stats([r[m] for r in all_resting])
            result['all_frames'][m]=stats([f[m] for f in all_frames])

        if all_valleys:
            for m in config['metrics']:
                vals=[v[m] for v in all_valleys]
                result['suggested_thresholds'][m]={
                    'ideal':round(percentile(vals,50),1),
                    'range_low':round(percentile(vals,10),1),
                    'range_high':round(percentile(vals,90),1),
                }
            print(f"\n  THRESHOLDS ({ex_key}):")
            print(f"  ┌──────────────────────┬──────────┬─────────────────────┐")
            print(f"  │ Metric               │  Ideal   │  Range              │")
            print(f"  ├──────────────────────┼──────────┼─────────────────────┤")
            for m in config['metrics']:
                st=result['suggested_thresholds'][m]
                print(f"  │ {m:<20} │ {st['ideal']:>6.1f}°  │ {st['range_low']:>5.1f}° – {st['range_high']:>5.1f}°        │")
            print(f"  └──────────────────────┴──────────┴─────────────────────┘")

        all_results[ex_key]=result

    elapsed=time.time()-t0
    output_path=os.path.expanduser("~/Desktop/exercise_analysis_real.json")
    with open(output_path,'w') as f:
        json.dump(all_results,f,indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Saved to: {output_path}")
    print(f"  ⏱ Total time: {elapsed:.1f}s")
    print(f"     Upload this file to Claude!")
    print(f"{'='*60}\n")

if __name__=='__main__':
    main()
