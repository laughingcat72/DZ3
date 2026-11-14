"""Build a self-contained notebook with full experiment code and cached result views."""
import ast
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Детекция и сегментация планктона без эталонной разметки

**RetinaNet · Faster R-CNN · GrabCut · SAM 2.1**

Работа выполнена на предоставленном наборе из 189 изображений. Исследование
аугментаций использует одну фиксированную выборку из 24 снимков для обеих моделей.
Обучение и дообучение не выполняются. Сохранённые таблицы, изображения и маски
позволяют проверить результаты без повторного инференса.

Четыре группы: **RetinaNet / оригиналы**, **RetinaNet / аугментации**,
**Faster R-CNN / оригиналы**, **Faster R-CNN / аугментации**.
В исходной формулировке после слов «четыре группы экспериментов» список отсутствует;
здесь явно принята факторная схема «2 модели × 2 условия».

**Важно:** COCO не содержит класса «планктон». Названия предсказанных классов
показываются буквально и не являются определением вида организма. Число рамок и
confidence не измеряют точность. mAP, Precision, Recall и истинный IoU масок не
рассчитываются: эталонной разметки нет.''')
md('''## 0. Запуск и протокол

При `RUN_EXPERIMENTS=False` ячейки читают уже сохранённые результаты.
Для полного пересчёта сначала установите зависимости по README, подключите архив
через `scripts/import_data.py`, скачайте SAM 2.1 и задайте `RUN_EXPERIMENTS=True`.
Полный исходный код ниже встроен в Notebook; те же функции находятся в `src/`.

Для каждой модели: FP32, batch=1, одинаковое ограничение длинной стороны 1024,
внутренний resize 640/1024, NMS=0.5, отчётный score≥0.25. Дополнительно сохранены
кандидаты score≥0.05 и числа рамок при 0.10/0.50. Три прогрева и три измерения
на каждую пару «снимок–вариант». Перед/после измерений CUDA синхронизируется.
Время включает подготовку тензора, передачу на GPU, преобразования модели,
forward, NMS и возврат на CPU; чтение файлов, аугментации и рисование исключены.
Порядок пар перемешан фиксированным seed и одинаков для обеих моделей.''')
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

Инвентаризация охватывает все 189 снимков: размеры, среднюю яркость, RMS-контраст,
диапазон P95–P5, доли отсечённых пикселей и неоднородность медиан по сетке 4×4.
Последний показатель — лишь пространственная характеристика изображения,
поскольку истинная маска фона отсутствует. Объекты могут влиять на эту оценку.

Выборка формируется **до инференса**: 12 кадров серии с цифровыми именами,
6 снимков серии Bosmina и 6 остальных. В каждой группе берутся равномерные
позиции по ранжированному RMS-контрасту. Это разведочная выборка с охватом
контрастности, а не случайная оценка всего датасета. Кадры одной серии могут
быть зависимы. Фрагменты `obj/class/conf` в именах не используются как разметка.''')
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

Реализовано шесть семейств: яркость (×1.15, ×1.35), контраст (×0.8, ×0.5
относительно среднего канала), независимый гауссов шум (σ=5, 20 в шкале 0–255),
гауссово размытие (σ=0.8, 2 пикселя), гамма (γ=0.7, 1.5), отражение и поворот
на 90° против часовой стрелки. Итого 12 вариантов + оригинал.
Шум детерминирован по seed, идентификатору снимка и имени преобразования.
Комбинированных преобразований нет, чтобы разделить их влияние.

Умеренные фотометрические изменения моделируют различия освещения. Сильное
осветление приводит к клиппингу; низкий контраст, шум и размытие скрывают
щетинки, прозрачные края и внутренние структуры. Гамма изменяет тональные
отношения и не является физической калибровкой микроскопа. Поворот на 90° и
отражение сохраняют пиксели без интерполяции и допустимы для задачи локализации
при произвольной ориентации препарата. Отражение нужно исключить, если важна
биологическая асимметрия. Масштабирование при подготовке до 1024 уже ограничивает
сохранение мелких признаков; метрические размеры без калибровки не оцениваются.''')
code('\n\n'.join(functions[n] for n in ['augment','augmentation_gallery']))
code('''if RUN_EXPERIMENTS: augmentation_gallery(sample)
display(pd.DataFrame([dict(variant=n, parameters=str(p)) for n,p in VARIANTS]))
for path in sorted((RESULTS / 'figures').glob('augmentations_*.jpg')):
    show_figure(path)''')
md('''## 3. Выбор детекторов и геометрическая устойчивость

**RetinaNet ResNet-50/FPN v2** — одноэтапный anchor-based детектор. На уровнях
пирамиды признаков плотные головы сразу предсказывают классы и уточнения рамок.
Focal Loss при исходном обучении уменьшает вклад многочисленных лёгких
отрицательных примеров. **Faster R-CNN ResNet-50/FPN v2** — двухэтапная модель:
RPN предлагает области, затем RoIAlign и отдельная голова классифицируют и
уточняют предложения. Одинаковое семейство backbone/FPN и одна библиотека
уменьшают различия, не связанные со схемой детекции. Рецепты предобучения
всё же различаются; сравнение не изолирует единственный архитектурный фактор.

Обе модели используют официальные `COCO_V1`, без обучения на этих снимках.
Выбор обусловлен воспроизводимостью, многомасштабными признаками и возможностью
честно сопоставить две схемы. Термин «современные методы» здесь реализован через
актуальные поддерживаемые реализации v2 и SAM 2.1; модели не объявляются SOTA
для микроскопии.

Перед сравнением рамки отражений и поворотов переводятся в координаты исходного
рабочего холста. Сопоставление class-agnostic, взаимно однозначное, IoU≥0.5;
оптимизация сначала максимизирует число допустимых пар, затем сумму IoU.
Считаются доля сохранившихся рамок, IoU совпавших рамок, смещение центров,
нормированное диагональю, и согласованность COCO-меток. При отсутствии исходных
рамок устойчивость **не определена**, а не равна 100%. Эти числа описывают
самосогласованность предсказаний, а не правильность обнаружения организма.''')
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
md('''## 4. Сегментация с одинаковыми box prompts

GrabCut моделирует распределения цветов переднего плана/фона и уточняет разрез
графа за пять итераций. Всё вне заданного прямоугольника считается фоном, поэтому
слишком тесная рамка необратимо обрезает выступающие структуры.
SAM 2.1 Hiera Tiny использует предобученный кодировщик изображения и декодер
масок с пространственной подсказкой. Используется оригинальная реализация Meta,
`multimask_output=False`, без дообучения. Расширение для CUDA-постобработки
отключено; удаление дыр и мелких компонент не применяется.

Основная демонстрационная схема: каждый второй снимок фиксированной выборки,
не более двух рамок Faster R-CNN с наибольшим score≥0.25. Выбор рамок смещён к
наиболее уверенным предсказаниям и не является оценкой всех объектов. Сам выбор
Faster R-CNN для этой демонстрации не доказывает его превосходства; итоговый
вывод опирается на таблицы и просмотр ошибок.

Оба метода получают **одни и те же целочисленные рамки**; оставляется граница
фона в один пиксель для GrabCut. Файл `segmentation_prompts.json` хранит и
запрошенную, и фактическую рамку. При использовании ручных рамок они помечены
`manual_demo`, имеют причину и анализируются отдельно. Для первых двух
автоматических рамок дополнительно исследуется сужение/расширение на 25%.
Два дополнительных случая `overlap_case` выбраны после визуального просмотра
на `img_041`, чтобы проверить объединение/разделение касающихся организмов.
Их статистика отделена от 22 основных рамок и четырёх вариантов их размера.
Всего сохранено 28 пар масок, без ручных подсказок.

Маски сохраняются отдельно для каждого экземпляра, в рабочем и исходном
разрешении. Последнее получается nearest-neighbor масштабированием и не
восстанавливает потерянные тонкие детали. Маски разных объектов могут
пересекаться. Наличие нескольких рамок не обеспечивает разделения перекрытий.

Время SAM включает кодирование всего изображения **для каждой рамки**, декодер
и получение маски на CPU; отдельно сохранены время encoder/decoder.
На практике embedding можно переиспользовать для нескольких рамок одного
снимка. GrabCut работает на CPU, SAM — на GPU; это сравнение выбранных
реализаций на данном оборудовании, а не чистой вычислительной сложности.''')
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
md('''## 5. Сводная таблица шести экспериментов

Четыре группы детекции и два метода сегментации представлены вместе.
Для сегментации сводка использует 22 основные одинаковые рамки; чувствительность
к размеру рамки и два выбранных после просмотра случая перекрытия приведены отдельно.''')
code("display(pd.read_csv(RESULTS / 'tables/experiment_comparison.csv').round(3))")
md('''## 5.1. Анализ ошибок и выводы

Следующий раздел составлен по сохранённым числам и визуально просмотренным
примерам. Из-за отсутствия разметки видимые пропуски/ложные обнаружения
указываются качественно, без приписывания точности всему датасету.''')
report=root/'REPORT.md'
if report.exists():md(report.read_text(encoding='utf-8'))
md('''## 6. Воспроизводимость и источники

Seed=20261112. В `results/environment.json` записаны фактические версии,
оборудование, параметры и границы замера времени. В `weights_manifest.json`
сохранены SHA256 весов; `dataset_inventory.csv` содержит SHA256 всех снимков.
Структурированные предсказания включают рамки в аугментированных, рабочих
исходных и нативных координатах. Время на другом устройстве будет отличаться;
полная побитовая идентичность CUDA между версиями не гарантируется.

- [Focal Loss / RetinaNet, Lin et al., 2017](https://arxiv.org/abs/1708.02002).
- [Faster R-CNN, Ren et al., 2015](https://arxiv.org/abs/1506.01497).
- [Torchvision: модели и веса](https://docs.pytorch.org/vision/0.21/models.html).
- [GrabCut: документация OpenCV](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html).
- [SAM 2: статья](https://arxiv.org/abs/2408.00714).
- [Официальный SAM 2 / инструкции и лицензия](https://github.com/facebookresearch/sam2).
- [Официальные веса SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny).

Результаты не доказывают пригодность к автоматическому подсчёту или определению
видов. Для количественной валидации потребуется небольшая независимая экспертная
разметка; полная ручная разметка не создавалась.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}})
nbf.write(nb,root/'plankton_experiments.ipynb')
print(root/'plankton_experiments.ipynb')
