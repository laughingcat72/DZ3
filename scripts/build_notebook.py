"""Build a self-contained notebook with full experiment code and cached result views."""
import ast
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Детекция и сегментация планктона

Цель — сравнить одноэтапный и двухэтапный предобученные детекторы, влияние аугментаций и сегментацию по полученным рамкам методами GrabCut и SAM 2.1.

Исследованы 189 изображений; для экспериментов выбраны 24 снимка. Четыре группы: RetinaNet и Faster R-CNN на оригиналах и аугментациях. Обучение моделей не проводится. Эталонной разметки нет: число обнаружений и confidence не трактуются как точность; COCO-классы не обозначают виды планктона.''')
md('''## 0. Запуск и параметры

`RUN_EXPERIMENTS=False` открывает сохранённые результаты. Для пересчёта установите зависимости и подключите данные по README, затем задайте `RUN_EXPERIMENTS=True`.

Обе модели: FP32, batch=1, длинная сторона рабочего изображения ≤1024, внутренний resize 640/1024, NMS=0.5, score≥0.25. Сохранены также кандидаты ≥0.05 и число рамок при порогах 0.10/0.50.

После трёх прогревов выполняются три замера; используется их медиана. CUDA синхронизируется до и после замера. Время включает подготовку тензора, передачу на GPU, обработку моделью, NMS и возврат на CPU; чтение файлов, аугментации и визуализация исключены. Порядок входов одинаков для обеих моделей, seed=20261112.''')
code('''from pathlib import Path
import os
RUN_EXPERIMENTS = False
DATA_DIR = Path(os.environ.get('PLANKTON_DATA_DIR', 'data'))
CHECKPOINT = Path(os.environ.get('PLANKTON_SAM_CHECKPOINT', 'checkpoints/sam2.1_hiera_tiny.pt'))
os.environ.setdefault('TORCH_HOME', str(Path('checkpoints/torch').resolve()))
from IPython.display import display, Image as DisplayImage, Markdown
def show_figure(path, width=1150):
    display(DisplayImage(filename=str(path), width=width))
''')

unpack_source=(root/'scripts/unpack_results.py').read_text(encoding='utf-8')
unpack_tree=ast.parse(unpack_source)
code(ast.get_source_segment(unpack_source,next(n for n in unpack_tree.body if isinstance(n,ast.FunctionDef)))+'\n\nif not RUN_EXPERIMENTS:\n    print("Restored image artifacts:", unpack_results(Path.cwd()))')
source=(root/'src/experiment.py').read_text(encoding='utf-8')
tree=ast.parse(source)
functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
prefix='\n'.join(source.splitlines()[:next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef))-1])
prefix=prefix.replace("ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()",'ROOT = Path.cwd()')
code(prefix)
code('\n\n'.join(functions[n] for n in ['setup','dump','read_rgb','resize_working','category','analyse_data']))
md('''## 1. Исходные данные

Набор содержит 180 JPEG, 8 BMP и 1 TIFF. Разрешение — 0.342–9.849 Мп, медиана 3.146 Мп; распространены 2048×1536 и 2592×1944. После рабочего resize RMS-контраст составляет 12.08–93.86, медиана 33.13; средняя яркость — 94.56–220.19 в шкале 0–255. EXIF-ориентация учитывается.

На `img_036` видны небольшие организмы на светлом неоднородном фоне; на `img_063` — плотное поле с касаниями; на `img_106` — крупный организм с длинными изогнутыми структурами; на `img_144` — прозрачное тело и тонкий отросток. Встречаются расфокусировка, мусор и волокна. Размеры объектов меняются от малой части кадра до почти всего изображения; физические размеры без калибровки не определяются.

До инференса выбраны 12 кадров серии с цифровыми именами, 6 Bosmina и 6 остальных, равномерно по ранжированному RMS-контрасту внутри групп. Выборка охватывает разные условия, но не является случайной оценкой всего набора. Фрагменты `obj/class/conf` в именах файлов не используются как разметка.''')
code('''setup()
if RUN_EXPERIMENTS:
    inventory, sample = analyse_data(DATA_DIR)
else:
    inventory = pd.read_csv(RESULTS / 'tables/dataset_inventory.csv')
    sample = pd.read_csv(RESULTS / 'tables/sample_manifest.csv')
display(inventory[['width','height','megapixels','mean_gray','rms_contrast','p95_minus_p5']].describe().round(2))
display(sample[['image_id','filename','group','rms_contrast']].round(2))
show_figure(RESULTS / 'figures/dataset_overview.jpg')
show_figure(RESULTS / 'figures/dataset_statistics.png')''')
md('''## 2. Аугментации

Используются яркость ×1.15/1.35, контраст ×0.8/0.5 относительно среднего канала, гауссов шум σ=5/20, размытие σ=0.8/2 пикселя, гамма 0.7/1.5, отражение и поворот на 90° против часовой стрелки. Всего 12 вариантов и оригинал. Шум определяется фиксированным seed; преобразования применяются отдельно.

Умеренные изменения яркости и контраста моделируют различия освещения. Сильное осветление насыщает пиксели; снижение контраста, шум и размытие скрывают прозрачные края, щетинки и внутренние структуры. Гамма изменяет видимость тёмных и светлых деталей. Поворот и отражение сохраняют пиксели без интерполяции и допустимы для локализации при произвольной ориентации организма; отражение непригодно для анализа биологической асимметрии. Уменьшение до 1024 пикселей также может терять мелкие детали.''')
code('\n\n'.join(functions[n] for n in ['augment','augmentation_gallery']))
code('''if RUN_EXPERIMENTS: augmentation_gallery(sample)
display(pd.DataFrame([dict(variant=n, parameters=str(p)) for n,p in VARIANTS]))
for path in sorted((RESULTS / 'figures').glob('augmentations_*.jpg')):
    show_figure(path)''')
md('''## 3. Детекция

**RetinaNet ResNet-50/FPN v2** — одноэтапная модель: головы на пирамиде признаков сразу предсказывают классы и координаты рамок. Focal Loss при предобучении уменьшает вклад многочисленных простых отрицательных примеров.

**Faster R-CNN ResNet-50/FPN v2** — двухэтапная модель: RPN предлагает области, затем RoIAlign и отдельная голова классифицируют и уточняют их.

Выбраны реализации Torchvision с общим семейством backbone/FPN и официальными весами COCO_V1 для сопоставления двух схем с многомасштабными признаками. Рецепты предобучения различаются; дообучение на планктоне не выполняется.

Для сравнения устойчивости рамки после отражения и поворота переводятся обратно в координаты исходного рабочего изображения. Сопоставление взаимно однозначное, без учёта класса, при IoU≥0.5: сначала максимизируется число пар, затем сумма IoU. Измеряются доля сохранившихся рамок, IoU пар, смещение центров относительно диагонали и согласованность меток. Если исходных рамок нет, доля сохранения не определена.''')
code('\n\n'.join(functions[n] for n in ['inverse_boxes','iou_matrix','match_stability','draw_detection','sync','build_detector','run_detection','summarize_detection','environment_record']))
code('''if RUN_EXPERIMENTS:
    environment_record()
    for model_name in ['retinanet', 'fasterrcnn']:
        run_detection(DATA_DIR, sample, model_name)
    summarize_detection()
display(pd.read_csv(RESULTS / 'tables/detection_four_groups.csv').round(3))
display(pd.read_csv(RESULTS / 'tables/detection_by_augmentation.csv').round(3))
show_figure(RESULTS / 'figures/detection_comparison.png')''')
code('''# Show the same six baseline images for both detectors, including failures.
for image_id in sample.image_id.iloc[::4]:
    for model in ['retinanet','fasterrcnn']:
        show_figure(RESULTS / 'detections' / f'{model}_{image_id}_original.jpg', width=750)''')
code('''# A noise stress test: more detections can include full-frame false regions.
for model in ['retinanet','fasterrcnn']:
    for variant in ['original','noise_20']:
        show_figure(RESULTS / 'detections' / f'{model}_img_030_{variant}.jpg', width=750)''')
md('''## 4. Сегментация

GrabCut моделирует цвета объекта и фона и уточняет разрез графа за пять итераций. Область вне рамки считается фоном. SAM 2.1 Hiera Tiny использует кодировщик изображения и декодер масок с box prompt; `multimask_output=False`, без обучения и CUDA-постобработки.

Основные подсказки — до двух рамок Faster R-CNN с наибольшим score≥0.25 на каждом втором снимке выборки. Оба метода получают одинаковые целочисленные рамки с границей фона в один пиксель. Координаты и источник сохранены в `segmentation_prompts.json`.

Получены 22 основные пары масок, 4 варианта сужения/расширения рамки на 25% и 2 дополнительных случая перекрытия на `img_041`, выбранных после визуального просмотра (`overlap_case`). Дополнительные случаи анализируются отдельно. Ручных рамок нет.

Бинарные маски сохранены в рабочем и исходном разрешении; увеличение выполняется nearest-neighbor. Сохранены также исходные изображения, рамки и наложения. GrabCut работает на CPU, SAM — на GPU. Время SAM включает кодирование изображения для каждой рамки, декодирование и возврат маски на CPU; время encoder/decoder записано отдельно.''')
seg=(root/'src/segmentation.py').read_text(encoding='utf-8')
st=ast.parse(seg)
code('\n\n'.join(ast.get_source_segment(seg,n) for n in st.body if isinstance(n,ast.FunctionDef)))
code('''if RUN_EXPERIMENTS:
    segment(DATA_DIR, CHECKPOINT, Path('manual_prompts.json'))
display(pd.read_csv(RESULTS / 'tables/segmentation_summary.csv').round(3))
display(pd.read_csv(RESULTS / 'tables/segmentation_runs.csv').round(3))
# IoU below is agreement BETWEEN METHODS, not IoU against ground truth.
prompts = json.loads((RESULTS / 'segmentation_prompts.json').read_text(encoding='utf-8'))
for p in prompts:
    show_figure(RESULTS / 'segmentation' / f"{p['prompt_id']}_comparison.jpg")''')
md('''## 5. Сравнительная таблица

Четыре группы детекции и два метода сегментации: параметры, время и наблюдаемые различия. Сводка сегментации рассчитана по 22 одинаковым основным рамкам.''')
code("display(pd.read_csv(RESULTS / 'tables/experiment_comparison.csv').round(3))")
md('''## 6. Анализ результатов и выводы

### Детекция и аугментации

На оригиналах RetinaNet даёт 72 рамки (3.00 на снимок), Faster R-CNN — 96 (4.00). Медиана времени — 59.33 и 112.11 мс. На аугментациях — 2.92 и 3.65 рамки на снимок, 59.30 и 111.93 мс. Числа на снимок используются потому, что аугментированных входов в 12 раз больше.

При контрасте ×0.5 среднее число рамок падает до 1.96 и 2.46, доля сохранившихся исходных рамок — до 0.467 и 0.521. При яркости ×1.15 доля сохранения составляет 0.896 и 0.813; при размытии σ=2 — 0.602 и 0.537. Потеря контраста и размытие скрывают границы и тонкие признаки.

Шум σ=20 увеличивает число рамок RetinaNet на `img_030` с 1 до 7, включая области почти на весь кадр. У Faster R-CNN при этом шуме доля сохранения равна 0.515. Новые рамки могут быть ложными или повторными.

После обратного преобразования координат отражение сохраняет 0.645 исходных рамок RetinaNet и 0.760 Faster R-CNN; поворот — 0.584 и 0.652. Следовательно, геометрические преобразования также меняют предсказания. IoU сопоставленных пар следует рассматривать вместе с долей исчезнувших рамок.

Примеры на исходных изображениях:

- RetinaNet, `img_036`: две нижние области локализованы, верхний организм пропущен.
- RetinaNet, `img_106`: рамка охватывает крупный организм и выступы, хотя COCO-метка `cake` не имеет биологического смысла.
- Faster R-CNN, `img_041`: наряду с организмами выделены весь кадр и ложная область `kite` на фоне.
- На 12 полях с несколькими организмами Faster R-CNN выдаёт 13 рамок площадью ≥80% кадра, RetinaNet — 2. Такие рамки не разделяют отдельные экземпляры.

Для получения исходных областей выбран **RetinaNet**: время меньше примерно в 1.89 раза, а крупных фоновых рамок на рассмотренных полях меньше. Faster R-CNN местами находит дополнительные экземпляры и лучше сохраняет рамки при отдельных преобразованиях. Без разметки это не доказывает превосходства по точности.

### GrabCut и SAM 2.1

Медиана на основную рамку: GrabCut — 2455.87 мс, SAM — 115.75 мс. Медианы encoder/decoder SAM — 98.70/18.37 мс. Сравнение относится к указанным CPU/GPU и настройкам. Медиана IoU между масками — 0.849; это согласие методов, а не сравнение с эталоном.

- `prompt_000`, `img_036`: оба метода хорошо отделяют компактное тело от светлого фона, IoU=0.942.
- `prompt_001`: ложная рамка почти на весь снимок приводит к фоновой области у GrabCut и почти инверсной маске у SAM.
- `prompt_011`, `img_106`: SAM лучше отделяет длинный изогнутый выступ; GrabCut заполняет промежуток между ним и телом. У SAM остаются мелкие неточности.
- `prompt_014` и `prompt_016`: тонкий выступ частично искажён, некоторые прозрачные структуры пропущены. Причины — низкий контраст и тесная рамка.
- `prompt_022`: сужение рамки на 25% обрезает маску GrabCut; SAM сохраняет более полный контур за пределами подсказки.
- `prompt_025`: широкая ошибочная рамка даёт GrabCut 46.58 с на запрос; SAM работает быстрее, но также выделяет фон. Его внутренняя оценка около 0.991 не подтверждает правильность маски.
- `prompt_026`: рамка охватывает два соприкасающихся организма, оба метода объединяют их (IoU=0.963). В `prompt_027` рамка одного экземпляра помогает отделить его; GrabCut захватывает тонкий фрагмент у контакта, SAM даёт более компактную маску.

### Вывод

Предобученные детекторы дают начальные области для сегментации, но допускают пропуски, фоновые рамки и нестабильность при изменениях изображения. RetinaNet предпочтителен в проведённом сравнении по времени и наблюдаемым ошибкам крупных рамок. SAM 2.1 быстрее на использованной GPU и в отдельных случаях лучше сохраняет тонкие структуры; GrabCut подходит для компактных контрастных объектов. Ошибки рамки, низкий контраст и перекрытия остаются основными причинами неудач обоих сегментаторов. Полученные рамки и маски требуют визуальной проверки.''')
md('''## 7. Воспроизводимость и источники

Версии, оборудование и параметры записаны в `results/environment.json`; SHA256 весов — в `results/weights_manifest.json`, изображений — в `dataset_inventory.csv`. JSON содержит рамки на преобразованном снимке (`box_aug`), после обратного преобразования (`box_base`) и в исходном разрешении (`box_original`), формат `[x1,y1,x2,y2]`. Seed=20261112.

- [RetinaNet / Focal Loss](https://arxiv.org/abs/1708.02002).
- [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- [Torchvision: модели и веса](https://docs.pytorch.org/vision/0.21/models.html).
- [GrabCut / OpenCV](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html).
- [SAM 2: статья](https://arxiv.org/abs/2408.00714), [код](https://github.com/facebookresearch/sam2), [веса SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny).''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}})
nbf.write(nb,root/'plankton_experiments.ipynb')
print(root/'plankton_experiments.ipynb')
