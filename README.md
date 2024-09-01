# Боење на Црно-Бели Слики
Овој проект се фокусира на автоматско боење на црно-бели слики со користење на различни модели за машинско учење, вклучувајќи претходно тренирани модели, понатамошно тренирање на постоечки архитектури, и градење сопствени модели од почеток. Секој метод е тестиран под различни услови за споредба на нивната ефективност, прецизност и применливост.

## Содржина
- [Преглед на проектот](#pregled)
- [Предуслови](#preduslovi)
- [Инструкции за подесување](#instrukcii)
- [Извршување на секој дел](#izvrshuvanje)
    - [Претходно трениран модел со Caffe](#caffemodel)
    - [Дополнително тренирање со VGG-16](#vgg)
    - [Сопствен GAN модел со U-Net](#gan)
    - [VGG-19 со CUDA и PyTorch](#cuda)
    - [Сопствен модел со TensorFlow](#tenserflow)
- [Лиценца](#license)

## Преглед на проектот
Овој проект истражува различни методи за автоматско боење на црно-бели слики. Се споредуваат три главни пристапи:
- Користење на претходно тренирани модели.
- Дополнително тренирање на постоечки модели со користење на дополнителни бази на податоци.
- Имплементација на сопствени модели со архитектури како U-Net и GAN.

## Предуслови
- Python 3.x
- Jupyter Notebook или Google Colab
- Git
- PyTorch, TensorFlow, Keras, OpenCV
- GPU компатибилен со CUDA (за деловите со PyTorch и CUDA)
- симнување на CaffeModel потребни датотеки преку еден од двата линкови
  Официјален
  ```bash
  https://github.com/richzhang/colorization
  ```
  или
  ```bash
  https://github.com/dhananjayan-r/Colorizer/tree/master/models
  ```

## Инструкции за подесување
1. Клонирај го репозиториумот:
   ```bash
   git clone https://github.com/YourUsername/BoenjeCrnoBeliSliki.git
   ```
2. Влез во датотека каде се наоѓа проектот
   ```bash
   cd BoenjeCrnoBeliSliki
   ```
3. Импортирајте ги потребните библиотеки
4. Извршете употреба на соодветната платформа ( Google Colab / PyCharm )

## Извршување на секој дел

### Претходно трениран модел со Caffe
Овој метод користи претходно трениран модел од трудот „Colorful Image Colorization“ на Richard Zhang и соработниците, користејќи го Caffe framework и OpenCV.
Чекори за извршување:
- Отвори ја папката `CaffeModel`.
- Провери дали ги имаш потребните датотеки: `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, и `pts_in_hull.npy`.
- Изврши го Python скриптот:
  ```bash
  python caffe_colorization.py
  ```

### Дополнително тренирање со VGG-16
Овој дел користи VGG-16 архитектура за дополнително тренирање на бази на податоци како CIFAR-10, CIFAR-100 и Oxford-IIIT Pet.

Чекори за извршување:
- Отвори `Treniranje_PredTrenirani_Modeli.ipynb` во Google Colab.
- Обезбеди пристап до Базите на податоци потребни.
- Изврши ги сите чекори во бележникот `Treniranje_PredTrenirani_Modeli.ipynb` за да го тренираш моделот на избраната база на податоци.

### Сопствен GAN модел со U-Net
Имплементира сопствена Генеративна Противничка Мрежа (GAN) со U-Net архитектура.

Чекори за извршување:
- Отвори `Treniranje_PredTrenirani_Modeli.ipynb` во Google Colab.
- Обезбеди пристап до Базите на податоци потребни.
- Изврши ги сите чекори во бележникот `Treniranje_PredTrenirani_Modeli.ipynb` за да го тренираш моделот на избраната база на податоци.

### VGG-19 со CUDA и PyTorch
Овој метод се базира на GAN пристап со VGG-19 архитектура, користејќи CUDA за забрзано тренирање на NVIDIA графички картички.

Чекори за извршување:
- Одете во `VGG-ICUDA`.
- Обезбеди дека имаш CUDA-компатибилна графичка картичка и потребните драјвери.
- Отвори го `vgg-icuda-nvidia.py` во PyCharm или твојот омилен IDE за Python.
- Прилагоди ги патеките за базата на податоци и поставките за CUDA по потреба, потоа изврши:
  ```bash
  python vgg-icuda-nvidia.py
  ```
### Сопствен модел со TensorFlow
Едноставна Конволуциска Невронска Мрежа (CNN) користејќи TensorFlow и Keras за основно боење.

Чекори за извршување:
- Отвори ја папката `GoogleColabs`.
- Отвори го бележникот `Treniranje_Svoj_Tenserflow_Model.ipynb` во Google Colab.
- Изврши ги чекорите во бележникот за да го тренираш и тестираш моделот со користење на Oxford-IIIT Pet базата на податоци.

- ## Лиценца
- Овој проект е лиценциран под MIT лиценца - погледнете ја LICENSE датотеката за детали.
