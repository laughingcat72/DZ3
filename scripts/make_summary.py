from pathlib import Path
import pandas as pd

def make_summary(results_dir):
    r=Path(results_dir)
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
        ('retinanet', 'original'): '72 рамки; на 12 общих полях 2 рамки площадью >=80% кадра; пропуск верхнего организма img_036.',
        ('retinanet', 'augmented'): 'blur=2: число рамок остаётся 3.00/снимок, сохраняется 60.2%; noise=20 на img_030: 4 из 7 рамок почти на весь кадр.',
        ('fasterrcnn', 'original'): '96 рамок; на 12 общих полях 13 рамок площадью >=80% кадра; отдельная рамка пары img_041 используется в prompt_027.',
        ('fasterrcnn', 'augmented'): 'contrast=0.5: 2.46 рамки/снимок и 52.1% сохранения; поворот 90 градусов: 65.2% сохранения.',
        ('GrabCut', 'same_22_boxes'): 'prompt_011: фон между телом и выступом; prompt_000 -> prompt_022: площадь маски 6009 -> 4081 пикселей.',
        ('SAM2', 'same_22_boxes'): 'prompt_000 -> prompt_022: 5714 -> 4705 пикселей; prompt_001: ошибочно выделено 96.85% кадра.',
    }
    for row in rows:row['observed_differences']=observations[(row['method'],row['condition'])]
    frame=pd.DataFrame(rows)
    frame.to_csv(r/'tables/experiment_comparison.csv',index=False)
    return frame

if __name__=='__main__':
    make_summary(Path(__file__).resolve().parents[1]/'results')
