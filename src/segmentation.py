import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from experiment import ROOT, RESULTS, DEVICE, SEED, CONF, setup, read_rgb, resize_working, dump, sync


def clip_box(box, w, h):
    x1,y1,x2,y2=box

    x1,y1=max(1,int(np.floor(x1))),max(1,int(np.floor(y1)))
    x2,y2=min(w-1,int(np.ceil(x2))),min(h-1,int(np.ceil(y2)))
    if x2-x1<3 or y2-y1<3: raise ValueError('Degenerate prompt')
    return [x1,y1,x2,y2]


def make_prompts(sample, manual_path=None):
    predictions=json.loads((RESULTS/'predictions_fasterrcnn.json').read_text(encoding='utf-8'))
    prompts=[]

    for row in sample.iloc[::2].itertuples():
        preds=[p for p in predictions if p['image_id']==row.image_id and p['variant']=='original' and p['score']>=CONF]
        for p in sorted(preds,key=lambda p:-p['score'])[:2]:
            prompts.append(dict(image_id=row.image_id,source='fasterrcnn',box=p['box_base'],
                                detection_id=p['detection_id'],confidence=p['score'],coco_label=p['label'],variant='base'))
    if manual_path and Path(manual_path).exists():
        manual=json.loads(Path(manual_path).read_text(encoding='utf-8'))
        for p in manual:
            assert p.get('source')=='manual_demo' and p.get('reason'), 'Manual prompts must be explicitly identified.'
        prompts.extend(manual)

    for p in prompts[:2]:
        b=np.array(p['box']);center=(b[:2]+b[2:])/2;half=(b[2:]-b[:2])/2
        for name,factor in [('tight_0.75',.75),('loose_1.25',1.25)]:
            q={**p,'variant':name,'box':np.r_[center-half*factor,center+half*factor].tolist()}
            prompts.append(q)


    for model,det_id in [('retinanet',0),('fasterrcnn',4)]:
        candidates=json.loads((RESULTS/f'predictions_{model}.json').read_text(encoding='utf-8'))
        matches=[p for p in candidates if p['image_id']=='img_041' and p['variant']=='original' and p['detection_id']==det_id and p['score']>=CONF]
        if len(matches)==1:
            p=matches[0]
            prompts.append(dict(image_id=p['image_id'],source=model,box=p['box_base'],detection_id=det_id,
                                confidence=p['score'],coco_label=p['label'],variant='overlap_case',
                                selection_note='Post-hoc qualitative case chosen after visual inspection of touching organisms; not part of the base timing summary.'))
    for i,p in enumerate(prompts): p['prompt_id']=f'prompt_{i:03d}'
    return prompts


def grabcut(im, box):
    x1,y1,x2,y2=box
    mask=np.zeros(im.shape[:2],np.uint8)
    bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
    cv2.setRNGSeed(SEED)
    cv2.grabCut(im,mask,(x1,y1,x2-x1,y2-y1),bg,fg,5,cv2.GC_INIT_WITH_RECT)
    return (mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)


def overlay(im, mask, color=(0,195,135)):
    result=im.copy()
    result[mask]=(result[mask].astype(float)*.55+np.array(color)*.45).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result,contours,-1,(255,60,30),1)
    return result


def render_comparison(im, box, masks, timings, prompt, agreement, path):
    a,b=masks['GrabCut'],masks['SAM2']
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    views=[(im,'Original'),(im,'Identical box prompt'),(a,'GrabCut binary mask'),
           (b,'SAM 2.1 binary mask'),(overlay(im,a),f"GrabCut | {timings['GrabCut']:.0f} ms"),
           (overlay(im,b),f"SAM 2.1 | {timings['SAM2']:.0f} ms")]
    for ax,(view,title) in zip(axes.ravel(),views):
        ax.imshow(view,cmap='gray' if view.ndim==2 else None,vmin=0,vmax=1 if view.ndim==2 else 255)
        ax.set_title(title,fontsize=12,pad=9);ax.axis('off')
    x1,y1,x2,y2=box
    axes[0,1].add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='red',lw=2))
    agreement_text=f'{agreement:.3f}' if agreement is not None else 'NA'
    fig.suptitle(f"{prompt['prompt_id']} | {prompt['image_id']} | {prompt['source']} | {prompt['variant']} | agreement IoU={agreement_text}",fontsize=13)
    fig.savefig(path,dpi=125);plt.close(fig)


def segment(data_dir, checkpoint, manual_path=None, resume=False):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sample=pd.read_csv(RESULTS/'tables/sample_manifest.csv')
    prompts=make_prompts(sample,manual_path)
    if not prompts:raise RuntimeError('No detector prompts; explicitly supply manual_demo boxes.')
    paths={p.name:p for p in Path(data_dir).rglob('*') if p.is_file()}
    model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device=DEVICE,apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
    rows=[];used=[];previous={}
    if resume and (RESULTS/'tables/segmentation_runs.csv').exists():
        rows=pd.read_csv(RESULTS/'tables/segmentation_runs.csv').to_dict('records')
        previous={p['prompt_id']:p for p in json.loads((RESULTS/'segmentation_prompts.json').read_text())}

    first=resize_working(read_rgb(paths[sample.iloc[0].filename]))
    with torch.inference_mode():
        predictor.set_image(first)
        predictor.predict(box=np.array([1,1,first.shape[1]-1,first.shape[0]-1]),multimask_output=False)
    for i,p in enumerate(prompts):
        row=sample[sample.image_id==p['image_id']].iloc[0]
        original=read_rgb(paths[row.filename]);im=resize_working(original);h,w=im.shape[:2]
        box=clip_box(p['box'],w,h);p={**p,'box_requested':p['box'],'box':box,'working_width':w,'working_height':h}
        used.append(p)
        existing=[r for r in rows if r['prompt_id']==p['prompt_id']]
        if existing:
            assert len(existing)==2 and {r['method'] for r in existing}=={'GrabCut','SAM2'}
            old=previous[p['prompt_id']]
            assert all(old[k]==p[k] for k in ['image_id','source','variant','box']), 'Cannot resume with changed prompts'
            continue
        masks={};timings={};errors={}
        for method in ['GrabCut','SAM2']:
            ts=[];encoder_times=[];decoder_times=[];quality=None
            for repeat in range(3):
                sync();start=time.perf_counter()
                try:
                    if method=='GrabCut': mask=grabcut(im,box)
                    else:
                        with torch.inference_mode():

                            predictor.set_image(im);sync();encoded=time.perf_counter()
                            out,quality,_=predictor.predict(box=np.array(box),multimask_output=False)
                            sync();decoded=time.perf_counter()
                        mask=out[0].astype(bool)
                        encoder_times.append((encoded-start)*1000);decoder_times.append((decoded-encoded)*1000)
                    sync();ts.append((time.perf_counter()-start)*1000)
                except (cv2.error,RuntimeError) as e:
                    errors[method]=str(e);mask=np.zeros((h,w),bool);ts.append((time.perf_counter()-start)*1000)
                    break
            masks[method]=mask
            timings[method]=float(np.median(ts))
            mask_name=f"{p['prompt_id']}_{method.lower()}"
            Image.fromarray(mask.astype(np.uint8)*255).save(RESULTS/'masks'/f'{mask_name}.png')
            native=cv2.resize(mask.astype(np.uint8),(original.shape[1],original.shape[0]),interpolation=cv2.INTER_NEAREST)
            Image.fromarray(native*255).save(RESULTS/'masks'/f'{mask_name}_native.png')
            Image.fromarray(overlay(im,mask)).save(RESULTS/'segmentation'/f'{mask_name}_overlay.jpg',quality=90)
            x1,y1,x2,y2=box
            inside=np.zeros_like(mask);inside[y1:y2,x1:x2]=True
            rows.append(dict(prompt_id=p['prompt_id'],image_id=p['image_id'],source=p['source'],prompt_variant=p['variant'],
                             method=method,median_ms=timings[method],min_ms=min(ts),max_ms=max(ts),repeats=len(ts),
                             encoder_ms=float(np.median(encoder_times)) if encoder_times else None,
                             decoder_ms=float(np.median(decoder_times)) if decoder_times else None,
                             mask_area_px=int(mask.sum()),area_fraction=float(mask.mean()),
                             foreground_outside_box=int((mask&~inside).sum()),
                             sam_predicted_iou=float(np.ravel(quality)[0]) if quality is not None else None,
                             error=errors.get(method,''),box=json.dumps(box)))
        a,b=masks['GrabCut'],masks['SAM2'];union=(a|b).sum()
        agreement=float((a&b).sum()/union) if union else None
        for r in rows[-2:]:r['method_agreement_iou']=agreement
        render_comparison(im,box,masks,timings,p,agreement,RESULTS/'segmentation'/f"{p['prompt_id']}_comparison.jpg")
        print(f"Segmentation {i+1}/{len(prompts)}: {p['image_id']} {p['source']} {p['variant']}",flush=True)
        pd.DataFrame(rows).to_csv(RESULTS/'tables/segmentation_runs.csv',index=False)
    dump(used,RESULTS/'segmentation_prompts.json')
    df=pd.DataFrame(rows)
    df.groupby(['method','source','prompt_variant']).agg(prompts=('prompt_id','size'),median_ms=('median_ms','median'),
       empty_masks=('mask_area_px',lambda s:int((s==0).sum())),mean_area_fraction=('area_fraction','mean'),
       median_agreement_iou=('method_agreement_iou','median')).reset_index().to_csv(RESULTS/'tables/segmentation_summary.csv',index=False)
    dump(dict(model='SAM 2.1 Hiera Tiny',config='configs/sam2.1/sam2.1_hiera_t.yaml',
       checkpoint=Path(checkpoint).name,device=DEVICE,precision='float32',multimask_output=False,
       postprocessing=False,grabcut_iterations=5,repeats=3,
       timing='GrabCut: CPU, full working image. SAM2: GPU when available, includes set_image encoder for EACH box plus prompt decoder and mask transfer. Image IO/plotting excluded. Encoder can be reused for boxes of the same image.'),RESULTS/'segmentation_config.json')
    del model,predictor;gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manual',type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();setup();segment(args.data,args.checkpoint,args.manual,args.resume)
