import ast
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Детекция и сегментация планктона

Проверяются устойчивость рамок RetinaNet и Faster R-CNN при аугментациях и влияние этих рамок на маски GrabCut и SAM 2.1. Используются предобученные модели без дообучения.

Исследованы 189 снимков. Сравнение выполнено на 24 одинаковых снимках: 2 детектора × 13 вариантов = 624 обработки. Четыре группы: каждая модель на оригиналах и на 12 аугментациях. Эталонных рамок и масок нет; confidence, число обнаружений и согласие масок не считаются точностью. COCO-метки не обозначают виды организмов.''')
md('''## 0. Запуск и параметры

При `RUN_EXPERIMENTS=False` читаются сохранённые результаты. Для пересчёта: зависимости, данные и веса по README, затем `RUN_EXPERIMENTS=True`.

Параметры обеих моделей: FP32, batch=1, рабочая длинная сторона ≤1024, внутренний resize 640/1024, NMS=0.5, отчётный score≥0.25. Кандидаты сохранены от 0.05; число рамок дополнительно рассчитано при 0.10 и 0.50.

Три прогрева и три измерения на каждый вход; сравнивается медиана. CUDA синхронизируется до и после замера. Включены подготовка тензора, передача на GPU, обработка моделью, NMS и возврат на CPU. Чтение файлов, аугментации и рисование исключены. Порядок входов одинаков, seed=20261112. Оборудование: RTX 3060 Laptop, 6 ГБ; GrabCut использует CPU.''')
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

Набор содержит 180 JPEG, 8 BMP и 1 TIFF. Площадь снимка составляет от 0.342 до 9.849 Мп, медиана 3.146 Мп. RMS-контраст после рабочего resize: от 12.08 до 93.86, медиана 33.13. Средняя яркость: от 94.56 до 220.19 в шкале 0-255. EXIF-ориентация учитывается.

| Снимок | Исходное разрешение | RMS-контраст | Наблюдение |
|---|---|---:|---|
| `img_036` | 2048×1536 | 12.08 | Три небольших организма, светлый фон с мелкими пятнами |
| `img_063` | 2048×1536 | 50.47 | Плотное поле, касания и перекрытия |
| `img_106` | 1634×1440 | 22.39 | Крупный организм, серый фон, длинный изогнутый выступ |
| `img_144` | 2592×1944 | 17.04 | Прозрачное тело, тонкий отросток и слабо выраженные верхние структуры |

Размеры областей заметно различаются: рамка компактного организма в `prompt_000` занимает 102×78 пикселей рабочего кадра, в `prompt_011` рамка крупного экземпляра занимает 872×763. Это размеры предложенных областей, не эталонные размеры тел; калибровки для микрометров нет.

До инференса выбраны 12 кадров серии с цифровыми именами, 6 Bosmina и 6 остальных. В каждой группе взяты равномерные позиции по ранжированному RMS-контрасту. Выводы относятся к этой выборке. Фрагменты `obj/class/conf` в именах файлов не использованы как разметка.''')
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

| Преобразование | Уровни | Что проверяется и что может теряться |
|---|---|---|
| Яркость | ×1.15, ×1.35 | Различия освещения; насыщение светлых участков |
| Контраст относительно среднего канала | ×0.8, ×0.5 | Ослабление границ прозрачных тел |
| Гауссов шум | σ=5, 20 в шкале 0-255 | Устойчивость к помехам, видимость тонких структур |
| Гауссово размытие | σ=0.8, 2 пикселя | Расфокусировка, сглаживание щетинок и краёв |
| Гамма | 0.7, 1.5 | Изменение видимости светлых и тёмных деталей |
| Геометрия | Отражение; поворот 90° против часовой стрелки | Зависимость рамок от ориентации организма |

Всего 12 вариантов и оригинал. Преобразования применяются отдельно; шум определяется seed и идентификатором снимка. Умеренные фотометрические изменения допустимы для проверки условий съёмки. Сильное размытие и шум могут скрыть морфологические признаки. Поворот и отражение сохраняют пиксели без интерполяции; отражение не подходит для анализа биологической асимметрии. Рабочий resize до 1024 пикселей также ограничивает видимость мелких деталей.''')
code('\n\n'.join(functions[n] for n in ['augment','augmentation_gallery']))
code('''if RUN_EXPERIMENTS: augmentation_gallery(sample)
display(pd.DataFrame([dict(variant=n, parameters=str(p)) for n,p in VARIANTS]))
for path in sorted((RESULTS / 'figures').glob('augmentations_*.jpg')):
    show_figure(path)''')
md('''## 3. Детекция

RetinaNet ResNet-50/FPN v2 использует одноэтапную схему: головы на пирамиде признаков сразу предсказывают классы и координаты рамок. Focal Loss при предобучении уменьшает вклад многочисленных простых отрицательных примеров.

Faster R-CNN ResNet-50/FPN v2 работает в два этапа: RPN предлагает области, затем RoIAlign и отдельная голова классифицируют и уточняют их. Общее семейство backbone/FPN и реализация Torchvision позволяют сравнить две схемы с многомасштабными признаками. Веса у обеих моделей COCO_V1, но рецепты предобучения различаются.

Рамки поворотов и отражений сначала возвращаются в координаты исходного рабочего изображения. Сопоставление взаимно однозначное, без учёта класса, при IoU≥0.5: максимизируется число пар, затем сумма IoU. Измеряются доля сохранившихся рамок, IoU пар, смещение центров относительно диагонали и согласованность меток. Доля сохранения усредняется по снимкам; при отсутствии исходных рамок она не определена.''')
code('\n\n'.join(functions[n] for n in ['inverse_boxes','iou_matrix','match_stability','draw_detection','sync','build_detector','run_detection','summarize_detection','environment_record']))
code('''if RUN_EXPERIMENTS:
    environment_record()
    for model_name in ['retinanet', 'fasterrcnn']:
        run_detection(DATA_DIR, sample, model_name)
    summarize_detection()
display(pd.read_csv(RESULTS / 'tables/detection_four_groups.csv').round(3))
display(pd.read_csv(RESULTS / 'tables/detection_by_augmentation.csv').round(3))
show_figure(RESULTS / 'figures/detection_comparison.png')''')
code('''for image_id in sample.image_id.iloc[::4]:
    for model in ['retinanet','fasterrcnn']:
        show_figure(RESULTS / 'detections' / f'{model}_{image_id}_original.jpg', width=750)''')
code('''for model in ['retinanet','fasterrcnn']:
    for variant in ['original','noise_20']:
        show_figure(RESULTS / 'detections' / f'{model}_img_030_{variant}.jpg', width=750)''')
md('''## 4. Сегментация

GrabCut моделирует цвета объекта и фона и уточняет разрез графа за пять итераций. Всё вне рамки считается фоном. SAM 2.1 Hiera Tiny использует кодировщик изображения и декодер масок с box prompt; `multimask_output=False`, без обучения и CUDA-постобработки.

Оба метода получают одинаковые целочисленные рамки с границей фона в один пиксель. Основная выборка: до двух рамок Faster R-CNN с наибольшим score≥0.25 на каждом втором снимке. Схема дала 22 пары масок. Дополнительно: четыре изменения ширины и высоты рамки на ±25% и два случая перекрытия на `img_041`, выбранных после просмотра. Эти шесть случаев не входят в основной итог времени. Ручных рамок нет.

Координаты и источник каждой подсказки записаны в `segmentation_prompts.json`. Сохранены бинарные PNG в рабочем и исходном разрешении, рамки и наложения. Маски исходного размера получены nearest-neighbor увеличением. Время SAM включает кодирование изображения заново для каждой рамки, декодирование и возврат маски на CPU; encoder и decoder записаны отдельно.''')
seg=(root/'src/segmentation.py').read_text(encoding='utf-8')
st=ast.parse(seg)
code('\n\n'.join(ast.get_source_segment(seg,n) for n in st.body if isinstance(n,ast.FunctionDef)))
code('''if RUN_EXPERIMENTS:
    segment(DATA_DIR, CHECKPOINT, Path('manual_prompts.json'))
display(pd.read_csv(RESULTS / 'tables/segmentation_summary.csv').round(3))
display(pd.read_csv(RESULTS / 'tables/segmentation_runs.csv').round(3))
prompts = json.loads((RESULTS / 'segmentation_prompts.json').read_text(encoding='utf-8'))
for p in prompts:
    show_figure(RESULTS / 'segmentation' / f"{p['prompt_id']}_comparison.jpg")''')
md('''## 5. Сравнительная таблица

Четыре группы детекции и два метода сегментации. Для аугментаций единица обработки: один снимок с одним преобразованием. Для сегментации: одна рамка из 22 основных. В таблице указаны параметры, медианы времени и наблюдаемые ошибки.''')
summary_source=(root/'scripts/make_summary.py').read_text(encoding='utf-8')
summary_tree=ast.parse(summary_source)
summary_function=next(n for n in summary_tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_summary')
code(ast.get_source_segment(summary_source,summary_function)+"\n\nif RUN_EXPERIMENTS:\n    make_summary(RESULTS)\n\ndisplay(pd.read_csv(RESULTS / 'tables/experiment_comparison.csv').round(3))")
md('''## 6. Разбор результатов

### Какие рамки меняются при аугментациях

На оригиналах RetinaNet дал 72 рамки, Faster R-CNN 96. На один снимок приходится 3.00 и 4.00 рамки, медиана времени составляет 59.33 и 112.11 мс. На аугментациях получено 841 и 1052 рамки на 288 входов: 2.92 и 3.65 на вход, 59.30 и 111.93 мс.

| Преобразование | Рамок/снимок RetinaNet | Сохранено RetinaNet, % | Рамок/снимок Faster R-CNN | Сохранено Faster R-CNN, % |
|---|---:|---:|---:|---:|
| Яркость ×1.15 | 3.21 | 89.6 | 3.62 | 81.3 |
| Яркость ×1.35 | 3.33 | 65.3 | 3.58 | 67.1 |
| Контраст ×0.5 | 1.96 | 46.7 | 2.46 | 52.1 |
| Размытие σ=2 | 3.00 | 60.2 | 2.21 | 53.7 |
| Отражение | 3.33 | 64.5 | 3.96 | 76.0 |
| Поворот 90° | 2.83 | 58.4 | 4.62 | 65.2 |

Источник: `results/tables/detection_by_augmentation.csv`. «Сохранено» означает среднюю по снимкам долю исходных рамок, которым найдена пара с IoU≥0.5.

При размытии σ=2 RetinaNet по-прежнему выдаёт 3.00 рамки на снимок, но сохраняется только 60.2% исходных рамок. Поэтому одинаковое число обнаружений не означает одинаковую локализацию. При контрасте ×0.5 число рамок снижается до 1.96 и 2.46; доля сохранения падает ниже 53% у обеих моделей. После обратного преобразования координат поворот сохраняет 58.4% рамок RetinaNet и 65.2% Faster R-CNN: зависимость от ориентации остаётся.

На `img_030` шум σ=20 меняет выход RetinaNet с 1 до 7 рамок. Четыре из семи занимают от 96.3 до 97.9% кадра; их confidence лежит от 0.255 до 0.356. На изображении они обводят сцену целиком, а не отдельный организм. Рост числа рамок здесь включает ложные области.

### Пропуски, дубликаты и выбор детектора

- `img_036`: RetinaNet выделяет два нижних организма с confidence 0.767 и 0.515, верхний пропущен. Faster R-CNN выдаёт три рамки, но две занимают ≥80% кадра. Большее число рамок не устраняет этот пропуск.
- `img_144`: RetinaNet выдаёт пять рамок вокруг одного организма. У `detection_id=0` и `2` координаты совпадают: `[99.7,215.8,865.0,552.1]`, но метки различаются (`tennis racket` и `wine glass`). Ещё одна совпадающая пара имеет номера `1` и `4`. Такие выходы нельзя считать пятью организмами.
- `img_041`: Faster R-CNN даёт ложную область `kite` с confidence 0.521 и координатами `[94.3,668.8,124.3,691.4]`, а также рамку почти на весь кадр. При этом его локальная рамка левого организма в соприкасающейся паре полезна для сегментации (`prompt_027`).

Координаты выше приведены в пикселях рабочего изображения и округлены до 0.1; полные значения находятся в `predictions_*.json`. Визуализации доступны после распаковки в `results/detections/` по именам модели, снимка и варианта.

На 12 полях с несколькими организмами у RetinaNet 2 рамки площадью ≥80% кадра, у Faster R-CNN 13 (`box_diagnostics.csv`). Для начальных областей выбран RetinaNet: он примерно в 1.89 раза быстрее и реже обводит всё поле. Выбор не универсален: на соприкасающихся организмах `img_041` полезная отдельная рамка найдена Faster R-CNN.

### Как размер рамки влияет на маску

Для правого нижнего организма `img_036` сравниваются исходная рамка Faster R-CNN и её сужение. Методы получают одну и ту же рамку в каждой строке.

| Подсказка | Рамка [x1,y1,x2,y2] | Площадь GrabCut, пикс. | Площадь SAM, пикс. | IoU методов |
|---|---|---:|---:|---:|
| 000, исходная | [695, 500, 797, 578] | 6009 | 5714 | 0.942 |
| 022, ширина и высота ×0.75 | [708, 510, 784, 568] | 4081 | 4705 | 0.819 |

При сужении площадь маски GrabCut уменьшается на 32.1%, SAM на 17.7% относительно масок при исходной подсказке. У GrabCut видны прямые обрезанные края; SAM сохраняет 680 пикселей за пределами тесной рамки. Это чувствительность к подсказке, не измерение потери относительно эталона.

### Фон, тонкие структуры и перекрытия

| Случай | Измерение | Наблюдение на маске и наложении |
|---|---|---|
| `prompt_000`, компактное тело | IoU методов 0.942 | Оба метода отделяют тело от светлого фона |
| `prompt_001`, ошибочная большая рамка | Маска GrabCut: 50.51% кадра; SAM: 96.85% | GrabCut захватывает фон; SAM выделяет почти весь фон, оставляя организмы отверстиями. Внутренняя оценка SAM 0.990 не подтверждает правильность |
| `prompt_011`, длинный выступ | GrabCut: 318739 пикселей; SAM: 277068 | GrabCut заполняет промежуток между телом и изогнутым выступом. SAM отделяет выступ от фона, но оставляет мелкие фрагменты |
| `prompt_016`, прозрачное тело | IoU методов 0.919 | Оба сохраняют тело и хвост, но пропускают часть верхних прозрачных структур. Их часть расположена выше рамки |
| `prompt_026`, два соприкасающихся организма | Рамка `[589,479,851,732]`, IoU 0.963 | Оба метода объединяют пару в одну маску |
| `prompt_027`, левый организм той же пары | Рамка `[588,477,744,673]`, IoU 0.853 | Оба отделяют левый экземпляр; у GrabCut остаётся тонкий фрагмент в области контакта |

Источники: `segmentation_runs.csv`, `segmentation_prompts.json` и панели `prompt_..._comparison.jpg`, показанные выше. Высокое согласие в `prompt_026` сочетается с объединением двух организмов; оно не проверяет разделение экземпляров.

На 22 основных рамках медиана GrabCut составляет 2455.87 мс, SAM 115.75 мс. Для SAM медианы encoder и decoder равны 98.70 и 18.37 мс; это отдельно рассчитанные медианы. Сравниваются CPU-реализация GrabCut и GPU-реализация SAM. Медиана IoU между их масками составляет 0.849.

### Вывод

Для этой выборки RetinaNet выбран как более быстрый источник начальных областей с меньшим числом рамок почти на весь кадр. SAM 2.1 предпочтителен для первичных масок: он быстрее на использованной GPU, лучше отделяет выступ в `prompt_011` и менее чувствителен к сужению рамки в `prompt_022`. GrabCut подходит для компактного контрастного тела (`prompt_000`). Ошибочный выбор области и объединение соприкасающихся организмов не исправляются автоматически ни одним методом; результаты требуют проверки.''')
md('''## 7. Воспроизводимость и источники

Версии, оборудование и параметры: `results/environment.json`. SHA256 весов: `results/weights_manifest.json`; снимков: `dataset_inventory.csv`. JSON хранит `box_aug` на преобразованном снимке, `box_base` после обратного преобразования и `box_original` в исходном разрешении. Формат `[x1,y1,x2,y2]`, seed=20261112.

- [RetinaNet / Focal Loss](https://arxiv.org/abs/1708.02002).
- [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- [Torchvision: модели и веса](https://docs.pytorch.org/vision/0.21/models.html).
- [GrabCut / OpenCV](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html).
- [SAM 2: статья](https://arxiv.org/abs/2408.00714), [код](https://github.com/facebookresearch/sam2), [веса SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny).''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}})
nbf.write(nb,root/'plankton_experiments.ipynb')
print(root/'plankton_experiments.ipynb')
