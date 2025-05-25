# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Diabetes Mellitus (DM) adalah penyakit kronis serius yang terjadi ketika pankreas tidak menghasilkan cukup insulin, atau ketika tubuh tidak dapat secara efektif menggunakan insulin yang dihasilkannya. Kondisi ini menyebabkan peningkatan kadar glukosa dalam darah (hiperglikemia), yang jika tidak dikelola dengan baik, dapat menyebabkan komplikasi kesehatan jangka panjang yang parah, termasuk penyakit jantung, stroke, gagal ginjal, kebutaan, dan amputasi (World Health Organization, 2023). Menurut International Diabetes Federation (IDF), pada tahun 2021, sekitar 537 juta orang dewasa (20-79 tahun) hidup dengan diabetes di seluruh dunia, dan angka ini diperkirakan akan meningkat menjadi 643 juta pada tahun 2030 (International Diabetes Federation, 2021). Beban ekonomi dan sosial akibat diabetes sangat besar, mencakup biaya perawatan kesehatan yang tinggi serta hilangnya produktivitas. 

Pendeteksian dini diabetes merupakan kunci untuk mencegah atau menunda timbulnya komplikasi serius. Banyak individu yang menderita diabetes tidak menyadari kondisi mereka sampai penyakitnya sudah pada tahap lanjut, yang mempersulit intervensi efektif. Oleh karena itu, pengembangan sistem yang dapat memprediksi risiko diabetes pada individu sedini mungkin menjadi sangat krusial. Pendekatan ini memungkinkan intervensi medis dan perubahan gaya hidup yang lebih cepat, seperti diet dan olahraga, yang terbukti efektif dalam mengelola dan bahkan membalikkan pre-diabetes atau diabetes tipe 2 pada tahap awal (Knowler et al., 2002).

Riset sebelumnya oleh Butt, Letchmunan, Ali, Hassan, Baqir, & Sherazi (2021), berjudul "Machine Learning Based Diabetes Classification and Prediction for Healthcare Applications", telah menyoroti pentingnya data kesehatan yang sensitif untuk deteksi dini penyakit mematikan seperti diabetes. Penelitian tersebut mengusulkan pendekatan machine learning untuk klasifikasi dan prediksi diabetes stadium awal, bahkan mempertimbangkan sistem pemantauan berbasis IoT. Mereka menggunakan algoritma seperti Random Forest, Multilayer Perceptron (MLP), dan Logistic Regression untuk klasifikasi, serta LSTM, Moving Averages, dan Linear Regression untuk prediksi. Hasil mereka menunjukkan MLP mencapai akurasi 86.08% dan LSTM 87.26%, menekankan perlunya terus mengembangkan dan membandingkan algoritma machine learning untuk solusi prediksi diabetes yang optimal

Proyek ini bertujuan untuk mengatasi masalah ini melalui pemanfaatan teknik machine learning. Dengan menganalisis data rekam medis pasien yang mencakup berbagai faktor risiko seperti kadar glukosa, tekanan darah, indeks massa tubuh (BMI), usia, dan riwayat kehamilan, model machine learning dapat dilatih untuk mengidentifikasi pola yang mengindikasikan kemungkinan seseorang menderita diabetes. Pendekatan ini akan melibatkan tahapan akuisisi data, pemrosesan awal data (termasuk penanganan nilai hilang dan scaling), pembangunan model prediktif menggunakan berbagai algoritma klasifikasi, dan evaluasi kinerja model untuk memilih yang terbaik. Diharapkan, model yang dikembangkan dapat berfungsi sebagai alat bantu skrining yang efektif, membantu profesional kesehatan dalam mengidentifikasi individu berisiko tinggi dan mendorong tindakan pencegahan lebih awal

## Business Understanding

### Problem Statements

1. Bagaimanakah cara melakukan prediksi risiko diabetes pada individu menggunakan model Machine Learning berdasarkan fitur-fitur rekam medis seperti kehamilan, glukosa, tekanan darah, ketebalan kulit, insulin, BMI, fungsi silsilah diabetes, dan usia?
2. Model Machine Learning manakah yang paling optimal dalam memprediksi diabetes berdasarkan dataset PIMA Indian Diabetes?

### Goals

1. Membuat model Machine Learning untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan fitur-fitur klinis dan demografis yang tersedia
2. Mengevaluasi dan merekomendasikan algoritma Machine Learning terbaik untuk tugas prediksi diabetes, dengan fokus pada akurasi dan metrik performa klasifikasi lainnya

### Solution statements
1.  **Penggunaan Dataset PIMA Indian Diabetes:**
    Dataset ini mencakup fitur-fitur medis relevan seperti jumlah kehamilan, konsentrasi glukosa, tekanan darah, ketebalan kulit, insulin, BMI, fungsi silsilah diabetes, dan usia, dengan `Outcome` sebagai variabel target (1 untuk diabetes, 0 untuk tidak diabetes).

2.  **Pembuatan dan Evaluasi Model Machine Learning untuk Prediksi Diabetes:**
    Model Machine Learning akan dikembangkan dan dievaluasi menggunakan beberapa algoritma klasifikasi, meliputi XGBoost, Decision Tree, Random Forest, Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN). Kinerja masing-masing model akan diukur menggunakan metrik evaluasi klasifikasi standar seperti Accuracy, Precision, Recall, dan F1-Score. Model dengan kinerja terbaik akan diidentifikasi sebagai solusi yang paling efektif, dan akan dilakukan **optimasi *hyperparameter*** untuk peningkatan performa lebih lanjut.

## Data Understanding

### Sumber Data
Dataset yang digunakan dalam proyek ini adalah PIMA Indian Diabetes Database, yang dapat diunduh dari Kaggle melalui tautan berikut:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Informasi Umum Data
Dataset ini merupakan kumpulan data diagnostik yang digunakan untuk memprediksi apakah seorang pasien wanita keturunan Indian Pima menderita diabetes atau tidak, berdasarkan beberapa pengukuran diagnostik.

* **Jumlah Data (Jumlah Baris):** Dataset ini berisi **768 entri (observasi/pasien)**.
* **Jumlah Fitur/Kolom:** Dataset ini terdiri dari **9 kolom (fitur)**.
* **Tipe Data:** 2 kolom bertipe `float64` dan 7 kolom bertipe `int64`.
* **Ukuran Memori:** Dataset ini membutuhkan sekitar **54.1 KB** memori.
* **Kondisi Data Awal:**
    * **Nilai Non-Null:** Semua kolom pada dataset ini menunjukkan **768 entri non-null**. Namun, perlu dicatat bahwa beberapa kolom numerik seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` memiliki nilai minimum 0, yang secara kontekstual tidak valid dan akan diperlakukan sebagai nilai hilang pada tahap *data preprocessing*.
    * **Nilai Duplikat:** Tidak ditemukan adanya **nilai duplikat** dalam dataset ini (0 duplikasi).

### Uraian Variabel (Fitur)
Berikut adalah uraian detail dari setiap variabel atau fitur yang terdapat dalam dataset PIMA Indian Diabetes:

1.  **`Pregnancies`** (`int64`): Jumlah kehamilan yang dialami oleh pasien. Rentang nilai: 0 hingga 17.
2.  **`Glucose`** (`int64`): Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral. Rentang nilai: 0 hingga 199.
3.  **`BloodPressure`** (`int64`): Tekanan darah diastolik (mm Hg). Rentang nilai: 0 hingga 122.
4.  **`SkinThickness`** (`int64`): Ketebalan lipatan kulit trisep (mm). Rentang nilai: 0 hingga 99.
5.  **`Insulin`** (`int64`): Kadar insulin serum 2 jam (mu U/ml). Rentang nilai: 0 hingga 846.
6.  **`BMI`** (`float64`): Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2). Rentang nilai: 0.0 hingga 67.1.
7.  **`DiabetesPedigreeFunction`** (`float64`): Skor yang mengindikasikan kemungkinan diabetes berdasarkan riwayat keluarga. Rentang nilai: 0.078 hingga 2.420.
8.  **`Age`** (`int64`): Usia pasien dalam tahun. Rentang nilai: 21 hingga 81.
9.  **`Outcome`** (`int64`): Variabel target (dependen) yang menunjukkan diagnosis diabetes. Nilai: 1 (positif, penderita diabetes) atau 0 (negatif, bukan penderita diabetes).

### Exploratory Data Analysis

**Univariate Analysis**
![univariate outcome](images/outcome.png)

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Refrensi
* Butt, U. M., Letchmunan, S., Ali, M., Hassan, F. H., Baqir, A., & Sherazi, H. H. R. (2021). Machine learning based diabetes classification and prediction for healthcare applications. Journal of healthcare engineering, 2021(1), 9930985.
* World Health Organization (WHO). (2023). Diabetes: Key facts. Diakses dari https://www.who.int/news-room/fact-sheets/detail/diabetes
* Diabetes Prevention Program Research Group. (2002). Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin. New England journal of medicine, 346(6), 393-403.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

