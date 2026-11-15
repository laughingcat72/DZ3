# Детекция и сегментация планктона

Сравнение предобученных RetinaNet и Faster R-CNN на микроскопических изображениях без разметки, влияние аугментаций и сегментация по рамкам детекторов методами GrabCut и SAM 2.1. Обучение моделей не проводится.

## Содержание

- [plankton_experiments.ipynb](plankton_experiments.ipynb): полный код, анализ данных, эксперименты, визуализации, анализ ошибок и выводы. Результаты сохранены в ячейках.
- `src/`: код экспериментов; `scripts/`: подготовка данных и запуск.
- `results/`: CSV/JSON с параметрами, координатами рамок, confidence и временем обработки.
- `results_archives/`: изображения с рамками, бинарные маски, наложения и аугментации. Распаковка: `python scripts/unpack_results.py`.
- `requirements.txt`: зависимости; `requirements-lock.txt`: версии полного окружения.

Для просмотра сохранённых результатов запуск моделей не нужен. При выполнении Notebook с `RUN_EXPERIMENTS=False` архивы распаковываются автоматически.

## Данные и методы

[Учебный датасет](https://drive.google.com/drive/folders/1LT2NdDSULHiW3FwzT63ewgVO3pwRRhOW): 189 изображений. Для сравнения заранее выбраны 24 снимка с различным контрастом. Обе модели получают одинаковые оригиналы и 12 вариантов аугментаций: яркость, контраст, шум, размытие, гамма, отражение и поворот. Для фотометрических преобразований используются два уровня интенсивности.

RetinaNet ResNet-50/FPN v2: одноэтапный детектор; Faster R-CNN ResNet-50/FPN v2: двухэтапный. Оба используют веса COCO_V1. Длинная сторона рабочего изображения ≤1024, внутренний resize 640/1024, score≥0.25, NMS=0.5. Время: медиана трёх повторов после прогрева, FP32, batch=1.

GrabCut (5 итераций, CPU) и SAM 2.1 Hiera Tiny (GPU) получают одинаковые рамки. Сохранены 22 основные пары масок, 4 проверки размера рамки и 2 случая перекрытия, выбранных после просмотра. Ручные рамки не использовались.

## Результаты

| Эксперимент | Обработанных снимков | Рамок на снимок | Медиана времени, мс |
|---|---:|---:|---:|
| RetinaNet, оригиналы | 24 | 3.000 | 59.33 |
| RetinaNet, аугментации | 288 | 2.920 | 59.30 |
| Faster R-CNN, оригиналы | 24 | 4.000 | 112.11 |
| Faster R-CNN, аугментации | 288 | 3.653 | 111.93 |
| GrabCut | 22 одинаковые рамки |: | 2455.87 на рамку |
| SAM 2.1 | те же 22 рамки |: | 115.75 на рамку |

На 12 полях с несколькими организмами RetinaNet даёт 2 рамки площадью ≥80% кадра, Faster R-CNN 13. При размытии σ=2 число рамок RetinaNet остаётся 3.00 на снимок, но сохраняется лишь 60.2% исходных рамок. При сужении одной и той же подсказки площадь маски GrabCut уменьшается на 32.1%, SAM на 17.7%. В общей рамке двух соприкасающихся организмов оба сегментатора объединяют их. Координаты, примеры ошибок и обоснование выбора RetinaNet и SAM приведены в разделе 6 Notebook. Эталонной разметки нет; число рамок, confidence и согласие масок не считаются точностью.

## Запуск

Проверенное окружение: Python 3.12.13, Windows 11, NVIDIA RTX 3060 Laptop (6 ГБ), PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124. Команды для PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
$env:SAM2_BUILD_CUDA = "0"
python -m pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Для CPU замените индекс PyTorch на `https://download.pytorch.org/whl/cpu`. Для установки SAM нужен Git; CUDA-расширение отключено.

Скачайте исходный учебный архив по ссылке выше, затем выполните:

```powershell
python scripts/import_data.py --zip "ПУТЬ_К_АРХИВУ.zip" --output data
python scripts/download_weights.py --output checkpoints
python scripts/run_all.py --data data --checkpoints checkpoints
```

После расчёта выполните ячейки Notebook с `RUN_EXPERIMENTS=False`, чтобы обновить выводы. Для запуска экспериментов из Notebook задайте `RUN_EXPERIMENTS=True`. Другие пути задаются через `PLANKTON_DATA_DIR` и `PLANKTON_SAM_CHECKPOINT`.

Проверка сохранённых результатов: `python scripts/validate_results.py`. Проверка преобразований координат: `python -m unittest discover -s tests -v`.

Публикация учебных изображений и результатов разрешена. В репозитории размещены иллюстрации выборки; полный исходный архив подключается отдельно.
