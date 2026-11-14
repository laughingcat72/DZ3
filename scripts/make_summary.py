"""Combine measured detection and base segmentation summaries without inference."""
from pathlib import Path
import pandas as pd
r=Path(__file__).resolve().parents[1]/'results'
d=pd.read_csv(r/'tables/detection_four_groups.csv');s=pd.read_csv(r/'tables/segmentation_summary.csv');rows=[]
for x in d.itertuples():
    rows.append(dict(task='detection',method=x.model,condition=x.group,input_pairs=x.image_variant_pairs,
                     median_ms=x.median_ms,detected_boxes=x.total_detections,
                     parameters='score=0.25;NMS=0.5;FP32;long_side=1024;3 repeats'))
for x in s[s.prompt_variant=='base'].itertuples():
    rows.append(dict(task='segmentation',method=x.method,condition='same_22_boxes',input_pairs=x.prompts,
                     median_ms=x.median_ms,detected_boxes=None,
                     parameters='5 iterations CPU' if x.method=='GrabCut' else 'SAM2.1 Tiny GPU;encoder+decoder;FP32'))
observations={
 ('retinanet','original'):'Быстрее; встречаются пропуски мелких объектов и ошибочные COCO-метки.',
 ('retinanet','augmented'):'При contrast=0.5 меньше рамок; при noise=20 больше рамок, включая ложные большие области.',
 ('fasterrcnn','original'):'Больше рамок, но встречаются фоновые области почти на весь кадр.',
 ('fasterrcnn','augmented'):'Сильное размытие/шум снижают сохранность исходных рамок; поворот меняет предсказания.',
 ('GrabCut','same_22_boxes'):'Чувствителен к размеру рамки; может соединять выступ с фоном и ошибаться на прозрачных краях.',
 ('SAM2','same_22_boxes'):'В ряде примеров лучше сохраняет форму; неверная большая рамка может выделить фон.'}
for row in rows:row['observed_differences']=observations[(row['method'],row['condition'])]
pd.DataFrame(rows).to_csv(r/'tables/experiment_comparison.csv',index=False)
