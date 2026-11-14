"""Re-render layout only, using saved masks and measured times; no model inference."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import RESULTS,read_rgb,resize_working,setup
from segmentation import render_comparison
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);args=p.parse_args();setup()
sample=pd.read_csv(RESULTS/'tables/sample_manifest.csv').set_index('image_id')
rows=pd.read_csv(RESULTS/'tables/segmentation_runs.csv')
paths={p.name:p for p in args.data.rglob('*') if p.is_file()}
prompts=json.loads((RESULTS/'segmentation_prompts.json').read_text())
for p in prompts:
    im=resize_working(read_rgb(paths[sample.loc[p['image_id'],'filename']]))
    d=rows[rows.prompt_id==p['prompt_id']].set_index('method')
    masks={method:np.array(Image.open(RESULTS/'masks'/f"{p['prompt_id']}_{method.lower()}.png"))>0 for method in ['GrabCut','SAM2']}
    render_comparison(im,p['box'],masks,d.median_ms.to_dict(),p,float(d.method_agreement_iou.iloc[0]),RESULTS/'segmentation'/f"{p['prompt_id']}_comparison.jpg")
print(f'Re-rendered {len(prompts)} figures without inference.')
