# multiclassseg

В репозитории содержатся скрипты для обучения, валидации и экспорта в onnx двух архитектур для семантической сегментации.
Использованные модели:

- [EfficientUnet](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf)
- [ESNet](https://github.com/xiaoyufenfei/ESNet)

Репозиторий базируется на моей [работе](https://github.com/DmitriyKras/segmentation_models) по сегментационным моделям на PyTorch. Loss функции для сегментации взяты [отсюда](https://github.com/qubvel-org/segmentation_models.pytorch).

Перед использованием установить зависимости:

`pip install -r requirments.txt`

## Dataset preparation

Подготовка датасета осуществляется с помощью **prepare_dataset.py**. В файле необходимо указать путь к разархивированному датасету PASCAL-Person-Part и конечную директорию для подготовленного датасета.

## Training

**train.py** - скрипт для запуска обучения:

`python3 train.py --model [model name] --n-classes [number of classes] --loss [loss function name] --img [input image size] --batch-size [number of images per batch] --epochs [number of epochs to train for] --augment [whether to use augmentation] --save-period [period in epochs to save checkpoint] --patience [number of epochs to stop training if no improvement observed] --data [path to data.yaml with dataset structure] --lr [learning rate] --decay [coef for exponential lr decay]`

`--model [effunet, esnet]`
`--loss [dice, iou, crossentropy]`
`--augment [True, False]`

`--data` data.yaml содержит:

**path**: абсолютный путь до корня датасета

**train**: относительный путь до папки с изображениями в jpg для обучения

**val**: относительный путь до папки с изображениями в jpg для валидации

**test**: относительный путь до папки с изображениями в jpg для теста (нужен для `--task test` in `val.py`)

Директории с сегментационными масками должны находится в директориях **train** и **val** и иметь название **masks**. Например train/images и train/masks. Маски должны иметь то же название, что и соответствующее изображение, и иметь расширение **.npy**. Маски содержат индекс класса на соответствующих пикселах изображения.

Скрипт сохраняет веса в процессе обучения в **weights/modelname** и результаты в **logs/modelname**.

## Validation

**val.py** - скрипт для валидации/теста обученной модели:

`python3 val.py --model [model name] --n-classes [number of classes] --img [input image size] --batch-size [number of images per batch] --data [path to data.yaml with dataset structure] --task [val or test] --weights [path to pt weights]`

`--task [val, test]` - использует директории val или test из *data.yaml*.

Скрипт сохраняет результаты в **logs/modelname**.

## Export

**export.py** - скрипт для экспорта в ONNX:

`python3 export.py --model [model name] --n-classes [number of classes] --img [input image size] --weights [path to pt weights] --onnx-file [path to onnx file] --opset [opset version]`

## Results

Для эксперимента были обучены две модели *EfficientUNet* и *ESNet*. Параметры: DiceLoss, 50 эпох, batch 32, размер изображения (320, 320), количество классов - 7. Модели обучены для задачи мультиклассовой сегментации. В процессе валидации оценивалась mIoU для трех уровней вложенности иерархической структуры классов (поэтому 3 значения для каждой эпохи) без учета фона.

Команды для обучения:

`python3 train.py --model effunet --n-classes 7 --batch-size 32 --epochs 50 --data data/data.yaml --img 320 --loss dice`

`python3 train.py --model esnet --n-classes 7 --batch-size 32 --epochs 50 --data data/data.yaml --img 320 --loss dice`

Результаты представлены в таблице ниже:

| Модель              | mIoU (whole body) | mIoU (up and down body) | mIoU (body parts) | Веса |
|---------            | :---------------: | :---------------: | :---------------: | -----|
| EfficientUNet B0    | 0.393             | 0.253             | 0.181             | [ссылка](https://drive.google.com/file/d/1rmlujjSeSEGJQXZEboKzv3ImF3xb4uZv/view?usp=sharing) |
| ESNet               | 0.372             | 0.240             | 0.170             | [ссылка](https://drive.google.com/file/d/1jRaVothrh8Np6JcMDzx65mOqX7Fv1mh2/view?usp=sharing) |

Примеры кривых обучения и метрик на каждой эпохе представлены на изображениях ниже (для EfficientUNet).

![Learning curves](/assets/loss.png)
![Metrics](/assets/metrics.png)

Команда для инференса:

`python3 inference.py --model esnet --n-classes 7 --img 320 --weights weights/esnet/esnet_best.pt --video 0`

## Thoughts

**1.** В данной работе для иерархической сегментации используется обычная мультиклассовая сегментация, поскольку задача может быть легко к ней сведена. Однако в дальнейшем имеет смысл обрать внимание на использование иерархической структуры, как это предложено [здесь](https://arxiv.org/abs/2203.14335) и уже реализовано [тут](https://github.com/lingorX/HieraSeg). Результаты показывают, что такой подход позволяет увеличить mIoU на PASCAL-Person-Part, хоть и не значительно.

**2.** Для повышения точности имеет смысл использовать методы transfer learning и использовать предобученную на ImageNet backbone или даже целый сегментор, обученный на большом датасете, например COCO. Кроме того, можно поэкспериментировать с более тяжелыми версиями EfficientNet и потюнить гиперпараметры.

**3.** В проект имеет смысл добавить некоторые полезные вещи, которые в сумме требуют более сложной реализации. Например всевозможные проверки разрешения входного изображения, ресайз с учетом соотношения сторон и размера входного изображения, автоматический препроцесс датасета и соответствующее построение модели (как это сделано в ultralytics), большее количество метрик и возможность их выбора(Precision, Recall ...), возможность возобновления обучения с чекпоинта и т.д.