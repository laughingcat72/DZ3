"""Verify result completeness, coordinate frames, counts, and identical segmentation prompts."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from unpack_results import unpack_results

root=Path(__file__).resolve().parents[1];unpack_results(root);r=root/'results'
sample=pd.read_csv(r/'tables/sample_manifest.csv').set_index('image_id')
runs=pd.read_csv(r/'tables/detection_runs.csv')
assert len(sample)==24 and len(runs)==624
assert not runs.duplicated(['model','image_id','variant']).any()
assert (runs.groupby(['model','image_id']).size()==13).all()
assert np.isfinite(runs.inference_ms).all() and (runs.inference_ms>0).all()
for model in ['retinanet','fasterrcnn']:
    ps=json.loads((r/f'predictions_{model}.json').read_text())
    bykey={}
    for p in ps:bykey.setdefault((p['image_id'],p['variant']),[]).append(p)
    for row in runs[runs.model==model].itertuples():
        preds=bykey.get((row.image_id,row.variant),[])
        assert sum(p['score']>=.25 for p in preds)==row.n_detections
        assert abs(np.median([row.time_rep1_ms,row.time_rep2_ms,row.time_rep3_ms])-row.inference_ms)<1e-6
        assert (r/'detections'/f'{model}_{row.image_id}_{row.variant}.jpg').exists()
        for p in preds:
            b=np.array(p['box_aug']);assert (b[:2]>=-1e-3).all() and (b[2:]>=b[:2]).all()
            assert b[2]<=row.width+.01 and b[3]<=row.height+.01
            b=np.array(p['box_original']);s=sample.loc[row.image_id]
            assert b[2]<=s.width+.02 and b[3]<=s.height+.02
prompts=json.loads((r/'segmentation_prompts.json').read_text());seg=pd.read_csv(r/'tables/segmentation_runs.csv')
assert len(prompts)==28 and len(seg)==56
assert seg.error.isna().all() and (seg.repeats==3).all()
assert not seg.duplicated(['prompt_id','method']).any()
assert all(p['source']!='manual_demo' for p in prompts)
for p in prompts:
    rows=seg[seg.prompt_id==p['prompt_id']];assert set(rows.method)=={'GrabCut','SAM2'}
    assert all(json.loads(b)==p['box'] for b in rows.box)
    masks=[]
    for method in ['GrabCut','SAM2']:
        name=f"{p['prompt_id']}_{method.lower()}"
        a=np.array(Image.open(r/'masks'/f'{name}.png'))
        assert a.shape==(p['working_height'],p['working_width']) and set(np.unique(a))<={0,255}
        native=Image.open(r/'masks'/f'{name}_native.png');s=sample.loc[p['image_id']]
        assert native.size==(s.width,s.height)
        assert int((a>0).sum())==int(rows[rows.method==method].mask_area_px.iloc[0])
        masks.append(a>0)
    iou=(masks[0]&masks[1]).sum()/(masks[0]|masks[1]).sum()
    assert abs(iou-rows.method_agreement_iou.iloc[0])<1e-9
    assert (r/'segmentation'/f"{p['prompt_id']}_comparison.jpg").exists()
record={'passed':True,'dataset_images':189,'sample_images':24,'detection_pairs':len(runs),
        'segmentation_prompts':len(prompts),'mask_files':len(prompts)*4,'manual_prompts':0,
        'checks':['complete pairs','nonnegative box geometry','score counts','timing medians','same prompts','binary masks','native dimensions','mask areas','method agreement']}
(r/'validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
