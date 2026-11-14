from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
import torch
import torchvision
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()
RESULTS = ROOT / 'results'
SEED = 20261112
CONF = 0.25
LONG_SIDE = 1024
REPEATS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VARIANTS = [
    ('original', {}),
    ('brightness_1.15', {'brightness': 1.15}),
    ('brightness_1.35', {'brightness': 1.35}),
    ('contrast_0.8', {'contrast': 0.8}),
    ('contrast_0.5', {'contrast': 0.5}),
    ('noise_5', {'noise_sigma': 5}),
    ('noise_20', {'noise_sigma': 20}),
    ('blur_0.8', {'blur_sigma': 0.8}),
    ('blur_2', {'blur_sigma': 2.0}),
    ('gamma_0.7', {'gamma': 0.7}),
    ('gamma_1.5', {'gamma': 1.5}),
    ('hflip', {'geometry': 'hflip'}),
    ('rot90', {'geometry': 'rot90_ccw'}),
]


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for p in ['tables', 'figures', 'detections', 'masks', 'segmentation', 'originals']:
        (RESULTS / p).mkdir(parents=True, exist_ok=True)


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()


def resize_working(im):
    h, w = im.shape[:2]
    scale = min(1.0, LONG_SIDE / max(h, w))
    return cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def category(name):
    if name.startswith('202502'):
        return 'multi_object_fields'
    if name.startswith('Bosmina cf longirostris Sackhalin'):
        return 'bosmina_series'
    return 'other_specimens'


def analyse_data(data_dir):
    rows = []
    paths = sorted(p for p in Path(data_dir).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
    if len({p.name for p in paths}) != len(paths):
        raise ValueError('Dataset contains duplicate basenames; rename before running.')
    for i, path in enumerate(paths):
        im = read_rgb(path)
        small = resize_working(im)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(float)
        p5, p95 = np.percentile(gray, [5, 95])

        tile_medians = [np.median(t) for stripe in np.array_split(gray, 4) for t in np.array_split(stripe, 4, axis=1)]
        rows.append(dict(image_id=f'img_{i:03d}', filename=path.name, group=category(path.name),
                         width=im.shape[1], height=im.shape[0], megapixels=im.shape[0]*im.shape[1]/1e6,
                         mean_gray=float(gray.mean()), rms_contrast=float(gray.std()),
                         p95_minus_p5=float(p95-p5), spatial_median_std=float(np.std(tile_medians)),
                         clipped_dark_fraction=float((gray<=1).mean()), clipped_light_fraction=float((gray>=254).mean()),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'tables/dataset_inventory.csv', index=False)

    selected = []
    for group, n in [('multi_object_fields', 12), ('bosmina_series', 6), ('other_specimens', 6)]:
        part = df[df.group == group].sort_values(['rms_contrast', 'filename'])
        selected.append(part.iloc[np.linspace(0, len(part)-1, min(n,len(part))).round().astype(int)])
    sample = pd.concat(selected).reset_index(drop=True)
    sample.to_csv(RESULTS / 'tables/sample_manifest.csv', index=False)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for ax, row in zip(axes.ravel(), sample.itertuples()):
        im = resize_working(read_rgb(next(p for p in paths if p.name == row.filename)))
        Image.fromarray(im).save(RESULTS / 'originals' / f'{row.image_id}.jpg', quality=92)
        ax.imshow(im); ax.set_title(f'{row.image_id} | {row.width}x{row.height}\nRMS={row.rms_contrast:.1f}', fontsize=9); ax.axis('off')
    fig.suptitle('Fixed sample: 24 / 189 images; contrast-stratified within three source groups')
    fig.tight_layout(); fig.savefig(RESULTS / 'figures/dataset_overview.jpg', dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(df.megapixels, bins=15); axes[0].set_xlabel('Image area, megapixels')
    axes[1].hist(df.rms_contrast, bins=15); axes[1].set_xlabel('Grayscale RMS contrast, 0..255')
    axes[2].scatter(df.mean_gray, df.spatial_median_std, s=15); axes[2].set_xlabel('Mean intensity'); axes[2].set_ylabel('Spatial median std (proxy)')
    fig.tight_layout(); fig.savefig(RESULTS / 'figures/dataset_statistics.png', dpi=140); plt.close(fig)
    return df, sample


def augment(im, name, image_id):
    params = dict(VARIANTS)[name]
    x = im.astype(np.float32)
    if 'brightness' in params: x *= params['brightness']
    if 'contrast' in params: x = (x - x.mean(axis=(0,1), keepdims=True)) * params['contrast'] + x.mean(axis=(0,1), keepdims=True)
    if 'noise_sigma' in params:
        seed = int.from_bytes(hashlib.sha256(f'{SEED}:{image_id}:{name}'.encode()).digest()[:8], 'little')
        x += np.random.default_rng(seed).normal(0, params['noise_sigma'], x.shape)
    if 'blur_sigma' in params: x = cv2.GaussianBlur(x, (0,0), params['blur_sigma'])
    if 'gamma' in params: x = 255.0 * np.power(x/255.0, params['gamma'])
    if name == 'hflip': x = x[:, ::-1]
    if name == 'rot90': x = np.rot90(x)
    return np.ascontiguousarray(np.clip(x, 0, 255).round().astype(np.uint8))


def augmentation_gallery(sample):

    for row in [sample.iloc[4], sample.iloc[15], sample.iloc[-1]]:
        im = read_rgb(RESULTS / 'originals' / f'{row.image_id}.jpg')
        fig, axes = plt.subplots(3, 5, figsize=(16, 10))
        for ax, (name, _) in zip(axes.ravel(), VARIANTS):
            ax.imshow(augment(im, name, row.image_id)); ax.set_title(name); ax.axis('off')
        for ax in axes.ravel()[len(VARIANTS):]: ax.axis('off')
        fig.suptitle(f'Augmentations | {row.image_id}'); fig.tight_layout()
        fig.savefig(RESULTS / 'figures' / f'augmentations_{row.image_id}.jpg', dpi=120); plt.close(fig)
    dump(dict(VARIANTS), RESULTS / 'augmentation_parameters.json')


def inverse_boxes(boxes, variant, width, height):

    b = np.asarray(boxes, dtype=float).reshape(-1,4).copy()
    if variant == 'hflip':
        b = b[:, [2,1,0,3]] * np.array([-1,1,-1,1]) + np.array([width,0,width,0])
    elif variant == 'rot90':

        b = np.stack([width-b[:,3], b[:,0], width-b[:,1], b[:,2]], axis=1)
    return b


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1,4); b = np.asarray(b, float).reshape(-1,4)
    lo = np.maximum(a[:,None,:2], b[None,:,:2]); hi = np.minimum(a[:,None,2:], b[None,:,2:])
    inter = np.maximum(hi-lo,0).prod(axis=2)
    area_a = np.maximum(a[:,2:]-a[:,:2],0).prod(axis=1)
    area_b = np.maximum(b[:,2:]-b[:,:2],0).prod(axis=1)
    return inter / np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-9)


def match_stability(original, augmented, width, height):
    a = [p for p in original if p['score']>=CONF]; b = [p for p in augmented if p['score']>=CONF]
    if not a or not b:
        return dict(baseline_count=len(a), augmented_count=len(b), matched=0,
                    retained_fraction=(0.0 if a else None), mean_matched_iou=None,
                    center_shift_diagonal=None, label_agreement=None)
    m = iou_matrix([p['box_base'] for p in a], [p['box_base'] for p in b])

    valid = m >= 0.5
    objective = valid.astype(float) * (min(m.shape)+1) + np.where(valid,m,0)
    ri, ci = linear_sum_assignment(-objective)
    pairs = [(int(i),int(j)) for i,j in zip(ri,ci) if valid[i,j]]
    shifts = [np.linalg.norm((np.array(a[i]['box_base'][:2])+a[i]['box_base'][2:])/2 -
                             (np.array(b[j]['box_base'][:2])+b[j]['box_base'][2:])/2) / np.hypot(width,height) for i,j in pairs]
    return dict(baseline_count=len(a), augmented_count=len(b), matched=len(pairs),
                retained_fraction=len(pairs)/len(a), mean_matched_iou=float(np.mean([m[i,j] for i,j in pairs])) if pairs else None,
                center_shift_diagonal=float(np.mean(shifts)) if pairs else None,
                label_agreement=float(np.mean([a[i]['label_id']==b[j]['label_id'] for i,j in pairs])) if pairs else None)


def draw_detection(im, preds, title, path):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(im)
    kept = [p for p in preds if p['score']>=CONF]
    for p in kept:
        x1,y1,x2,y2=p['box_aug']
        ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='#f04438',lw=1.5))
        ax.text(x1,y1, f"{p['label']} {p['score']:.2f}", color='white', fontsize=7,
                bbox=dict(facecolor='#a00000',alpha=.8,edgecolor='none'))
    ax.set_title(f'{title} | n={len(kept)} | threshold={CONF}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(path,dpi=110); plt.close(fig)


def sync():
    if DEVICE == 'cuda': torch.cuda.synchronize()


def build_detector(name):
    common = dict(min_size=640, max_size=1024)
    if name == 'retinanet':
        w = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=w, score_thresh=.05, nms_thresh=.5,
                                         detections_per_img=100, **common)
    elif name == 'fasterrcnn':
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=.05, box_nms_thresh=.5,
                                          box_detections_per_img=100, **common)
    else: raise ValueError(name)
    return model.to(DEVICE).eval(), w


def run_detection(data_dir, sample, model_name):
    model, weights = build_detector(model_name)
    paths = {p.name:p for p in Path(data_dir).rglob('*') if p.is_file()}
    first = resize_working(read_rgb(paths[sample.iloc[0].filename]))
    warm = torch.from_numpy(first).permute(2,0,1).float().div(255).to(DEVICE)
    with torch.inference_mode():
        for _ in range(3): model([warm])
    sync()
    del warm
    raw, runs, stability = [], [], []
    tasks = [(r, name) for r in sample.itertuples() for name,_ in VARIANTS]
    random.Random(SEED).shuffle(tasks)
    by_image = {}
    for step, (row, name) in enumerate(tasks):
        im0 = resize_working(read_rgb(paths[row.filename]))
        h,w = im0.shape[:2]
        im = augment(im0, name, row.image_id)
        times=[]
        for repeat in range(REPEATS):
            sync(); start=time.perf_counter()
            tensor=torch.from_numpy(im).permute(2,0,1).float().div(255).to(DEVICE)
            with torch.inference_mode(): output=model([tensor])[0]
            result={k:v.detach().cpu().numpy() for k,v in output.items()}
            sync(); times.append((time.perf_counter()-start)*1000)
        boxes = result['boxes']; back=inverse_boxes(boxes,name,w,h)
        preds=[]
        for idx,(box,base,score,label) in enumerate(zip(boxes,back,result['scores'],result['labels'])):
            p=dict(image_id=row.image_id, model=model_name, variant=name, detection_id=idx,
                   score=float(score), label_id=int(label), label=weights.meta['categories'][int(label)],
                   box_aug=box.tolist(),box_base=base.tolist(),
                   box_original=(base*np.array([row.width/w,row.height/h,row.width/w,row.height/h])).tolist())
            preds.append(p); raw.append(p)
        by_image[(row.image_id,name)]=preds
        runs.append(dict(image_id=row.image_id,model=model_name,variant=name,group='original' if name=='original' else 'augmented',
                         width=im.shape[1],height=im.shape[0],n_detections=sum(p['score']>=CONF for p in preds),
                         inference_ms=float(np.median(times)),inference_min_ms=min(times),inference_max_ms=max(times),
                         time_rep1_ms=times[0],time_rep2_ms=times[1],time_rep3_ms=times[2],
                         n_at_010=sum(p['score']>=.1 for p in preds),n_at_050=sum(p['score']>=.5 for p in preds)))
        draw_detection(im,preds,f'{model_name} | {row.image_id} | {name}',
                       RESULTS/'detections'/f'{model_name}_{row.image_id}_{name}.jpg')
        if step%12==0: print(f'{model_name}: {step+1}/{len(tasks)}',flush=True)

        if step%24==0:
            pd.DataFrame(runs).to_csv(RESULTS/'tables'/f'runs_{model_name}.csv',index=False)
    for row in sample.itertuples():
        im=resize_working(read_rgb(paths[row.filename]));h,w=im.shape[:2]
        for name,_ in VARIANTS[1:]:
            stability.append(dict(image_id=row.image_id,model=model_name,variant=name,
                                  **match_stability(by_image[(row.image_id,'original')],by_image[(row.image_id,name)],w,h)))
    dump(raw,RESULTS/f'predictions_{model_name}.json')
    pd.DataFrame(runs).to_csv(RESULTS/'tables'/f'runs_{model_name}.csv',index=False)
    pd.DataFrame(stability).to_csv(RESULTS/'tables'/f'stability_{model_name}.csv',index=False)
    dump(dict(model=model_name,weights=str(weights),url=weights.url,
              parameters=sum(p.numel() for p in model.parameters()),device=DEVICE,
              min_size=640,max_size=1024,internal_score_threshold=.05,report_threshold=CONF,nms=.5,
              precision='float32',batch_size=1,warmup=3,repeats=REPEATS),RESULTS/f'model_{model_name}.json')
    del model, output, tensor
    gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()


def summarize_detection():
    runs=pd.concat([pd.read_csv(RESULTS/'tables'/f'runs_{n}.csv') for n in ['retinanet','fasterrcnn']])
    stability=pd.concat([pd.read_csv(RESULTS/'tables'/f'stability_{n}.csv') for n in ['retinanet','fasterrcnn']])
    summary=runs.groupby(['model','group']).agg(images=('image_id','nunique'),variants=('variant','nunique'),
              image_variant_pairs=('image_id','size'),total_detections=('n_detections','sum'),
              detections_per_image=('n_detections','mean'),zero_detection_fraction=('n_detections',lambda x:float((x==0).mean())),
              median_ms=('inference_ms','median'),p90_ms=('inference_ms',lambda x:x.quantile(.9)),mean_ms=('inference_ms','mean')).reset_index()
    summary.to_csv(RESULTS/'tables/detection_four_groups.csv',index=False)
    byvar=runs.groupby(['model','variant']).agg(detections_per_image=('n_detections','mean'),median_ms=('inference_ms','median')).reset_index()
    stabvar=stability.groupby(['model','variant']).agg(retained_fraction=('retained_fraction','mean'),
               mean_matched_iou=('mean_matched_iou','mean'),mean_center_shift=('center_shift_diagonal','mean'),
               matched=('matched','sum'),baseline_boxes=('baseline_count','sum')).reset_index()
    byvar.merge(stabvar,on=['model','variant'],how='left').to_csv(RESULTS/'tables/detection_by_augmentation.csv',index=False)
    runs.to_csv(RESULTS/'tables/detection_runs.csv',index=False)
    stability.to_csv(RESULTS/'tables/detection_stability.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    for model in ['retinanet','fasterrcnn']:
        s=byvar[byvar.model==model].set_index('variant').reindex([n for n,_ in VARIANTS])
        axes[0].plot(range(len(s)),s.detections_per_image,marker='o',label=model)
        t=stabvar[stabvar.model==model].set_index('variant').reindex([n for n,_ in VARIANTS[1:]])
        axes[1].plot(range(len(t)),t.retained_fraction,marker='o',label=model)
    for ax,names,ylabel in [(axes[0],[n for n,_ in VARIANTS],'Mean detections / image'),(axes[1],[n for n,_ in VARIANTS[1:]],'Retained baseline boxes (IoU >= 0.5)')]:
        ax.set_xticks(range(len(names)),names,rotation=60,ha='right');ax.set_ylabel(ylabel);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RESULTS/'figures/detection_comparison.png',dpi=150);plt.close(fig)
    return summary


def environment_record():
    import importlib.metadata as md
    names=['torch','torchvision','numpy','pandas','opencv-python','matplotlib','scipy','Pillow','SAM-2','hydra-core','nbformat','nbclient']
    packages={n:md.version(n) for n in names if md.packages_distributions()}
    record=dict(seed=SEED,python=platform.python_version(),platform=platform.platform(),device=DEVICE,
                gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,cuda=torch.version.cuda,
                cpu_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),packages=packages,
                timing='Synchronized wall clock; CPU uint8 -> float tensor, H2D, model preprocessing, forward, NMS, D2H. Excludes image IO, augmentations, figures and model loading.',
                selection='24 contrast-stratified images: 12 multi-object fields, 6 Bosmina series, 6 other specimens; no prediction-based selection.',
                sam2_commit='2b90b9f5ceec907a1c18123530e92e794ad901a4')
    dump(record,RESULTS/'environment.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--stage',choices=['analyse','retinanet','fasterrcnn','summarize'],required=True)
    args=parser.parse_args();setup()
    if args.stage=='analyse':
        _,sample=analyse_data(args.data);augmentation_gallery(sample);environment_record()
    elif args.stage=='summarize':print(summarize_detection().to_string(index=False))
    else:run_detection(args.data,pd.read_csv(RESULTS/'tables/sample_manifest.csv'),args.stage)
