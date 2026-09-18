# Deteksi Fraktur Tulang pada Citra X-Ray Menggunakan Metode Convolutional Neural Network (CNN)

## Deskripsi

Project ini membahas penerapan Convolutional Neural Network (CNN) untuk mendeteksi dan mengklasifikasikan fraktur tulang pada citra X-Ray. Dataset yang digunakan berasal dari Kaggle dengan dataset **Bone Fracture Detection Computer Vision Project**.

Pada project ini digunakan **MobileNetV2** sebagai model CNN dengan pendekatan transfer learning. Selain menggunakan data citra secara langsung sebagai baseline, dilakukan beberapa eksperimen preprocessing untuk melihat pengaruh peningkatan kualitas citra terhadap performa model.

## Dataset

Dataset yang digunakan adalah:

**Bone Fracture Detection Computer Vision Project**

Dataset memiliki beberapa kelas fraktur tulang, yaitu:

- elbow positive
- fingers positive
- forearm fracture
- humerus fracture
- humerus
- shoulder fracture
- wrist positive

Dari proses pengolahan data, digunakan **1.804 citra** dengan 6 kelas setelah kelas dengan label 3 dikeluarkan.

Pembagian data:

- Train: 1.262 citra
- Validation: 271 citra
- Test: 271 citra

## Metode

### 1. Data Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

- Resize citra menjadi 224 × 224 piksel
- Konversi citra menjadi grayscale
- Normalisasi nilai piksel
- Konversi kembali menjadi 3 channel agar sesuai dengan input MobileNetV2

Selain preprocessing dasar, dilakukan eksperimen menggunakan beberapa metode peningkatan citra:

- CLAHE dengan `clipLimit = 2.0`
- CLAHE dengan `tileGridSize = (2,2)`
- Adaptive Mean Thresholding
- Adaptive Gaussian Thresholding

### 2. Convolutional Neural Network

Model yang digunakan adalah **MobileNetV2** dengan bobot ImageNet sebagai pretrained model.

Arsitektur model terdiri dari:

- MobileNetV2 sebagai feature extractor
- Global Average Pooling
- Dense layer 128 neuron dengan aktivasi ReLU
- Dropout 0.3
- Dense output layer dengan aktivasi Softmax

Base model MobileNetV2 dibuat **non-trainable**, sehingga proses training hanya dilakukan pada layer tambahan.

### 3. Training

Model dilatih menggunakan:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 10
- Evaluation metric: Accuracy

## Eksperimen

Beberapa skenario preprocessing dibandingkan untuk mengetahui pengaruhnya terhadap performa model.

| Skenario | Preprocessing | Accuracy | Loss |
|----------|---------------|----------|------|
| Baseline | Preprocessing dasar | 83.03% | 0.4879 |
| B1 | CLAHE `clipLimit=2.0` | 86.72% | 0.3412 |
| B2 | CLAHE `tileGridSize=(2,2)` | 83.76% | 0.3958 |
| C1 | Adaptive Mean Threshold | 81.55% | 0.5438 |
| C2 | Adaptive Gaussian Threshold | 78.97% | 0.5843 |

## Evaluasi Model

Evaluasi dilakukan menggunakan:

- Accuracy
- Loss
- Classification Report
- Confusion Matrix

Hasil eksperimen menunjukkan bahwa setiap metode preprocessing menghasilkan performa yang berbeda pada model MobileNetV2. Perbandingan dilakukan berdasarkan nilai accuracy dan loss pada data testing.

## Tools dan Library

Project ini menggunakan:

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- KaggleHub

## Alur Project

1. Load dataset
2. Eksplorasi dan pengecekan struktur dataset
3. Exploratory Data Analysis (EDA)
4. Data preprocessing
5. Pembagian data train, validation, dan test
6. Penerapan preprocessing dasar
7. Eksperimen CLAHE
8. Eksperimen Adaptive Thresholding
9. Training MobileNetV2
10. Evaluasi model
11. Perbandingan hasil setiap skenario
12. Visualisasi hasil menggunakan confusion matrix dan grafik accuracy

## Hasil

Model dengan preprocessing **CLAHE menggunakan `clipLimit=2.0`** memperoleh nilai accuracy sebesar **86.72%** pada data testing, dibandingkan baseline sebesar **83.03%**.

Hasil tersebut digunakan untuk melihat pengaruh preprocessing citra terhadap kemampuan MobileNetV2 dalam melakukan klasifikasi citra X-Ray pada dataset yang digunakan.

## Author

- Kevin Foreman Hutahaen
- Kartika Nur Savira
- Arimby Deby Setyoningrum
