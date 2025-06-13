**Rangkuman Repository** <br>
Repositori ini berisi solusi dan implementasi latihan-latihan dalam buku karya Aurélien Géron, "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd edition."<br>
<br>

**Tujuan**<br>
tujuan dari repositori ini adalah untuk memenuhi nilai mata kuliah Machine Learning. Repositori ini mencakup implementasi kode untuk latihan yang ditemukan di akhir setiap bab, serta kode dari dalam bab-bab pada buku.<br>
<br>

### Part I: The Fundamentals of Machine Learning

* **Chapter 1: The Machine Learning Landscape**
    * [cite_start]**Ringkasan:** Pengenalan konsep-konsep inti Machine Learning, kategori utamanya (supervised, unsupervised, dll.), dan tantangan utama seperti overfitting dan underfitting.
    * [cite_start]**Teori Singkat:** Teori dasarnya adalah bahwa sebuah sistem dapat belajar dari pengalaman (data). [cite_start]Hal ini didefinisikan secara formal oleh Tom Mitchell: sebuah program komputer dikatakan belajar dari pengalaman E sehubungan dengan tugas T dan ukuran kinerja P, jika kinerjanya pada T, yang diukur dengan P, meningkat seiring dengan pengalaman E.

* **Chapter 2: End-to-End Machine Learning Project**
    * [cite_start]**Ringkasan:** Panduan praktis melalui sebuah proyek ML lengkap, mulai dari membingkai masalah dan mendapatkan data hingga fine-tuning dan meluncurkan model.
    * [cite_start]**Teori Singkat:** Teori yang mendasari bab ini adalah metodologi terstruktur untuk proyek machine learning. [cite_start]Ini melibatkan pemilihan ukuran kinerja yang tepat, seperti *Root Mean Square Error* (RMSE) untuk tugas regresi, untuk mengevaluasi dan mengoptimalkan model secara objektif.

* **Chapter 3: Classification**
    * [cite_start]**Ringkasan:** Pembahasan mendalam tentang tugas klasifikasi, metrik performa, dan algoritma umum menggunakan dataset MNIST sebagai studi kasus.
    * [cite_start]**Teori Singkat:** Teori evaluasi klasifikasi berpusat pada metrik di luar akurasi sederhana[cite: 119]. [cite_start]Konsep utamanya meliputi *confusion matrix* untuk menghitung *true/false positive/negative* [cite: 120][cite_start], *precision* dan *recall* untuk mengukur kinerja pada kelas yang tidak seimbang [cite: 121, 122][cite_start], dan kurva ROC untuk memvisualisasikan *trade-off* antara *true positive rate* dan *false positive rate*.

* **Chapter 4: Training Models**
    * [cite_start]**Ringkasan:** Menjelajahi berbagai teknik pelatihan model, termasuk Regresi Linier, *Gradient Descent* (*Batch*, *Stochastic*, *Mini-batch*), Regresi Polinomial, dan model yang diregularisasi seperti *Ridge*, *Lasso*, dan *Elastic Net*.
    * [cite_start]**Teori Singkat:** Teori pelatihan model adalah tentang meminimalkan *cost function*. [cite_start]Untuk model linier, ini dapat dicapai secara langsung melalui persamaan matematis (*Normal Equation*)  [cite_start]atau secara iteratif melalui *Gradient Descent*, yang secara bertahap menyesuaikan parameter model untuk menemukan nilai minimum dari *cost function*. [cite_start]Regularisasi (seperti *Ridge* dan *Lasso*) secara teoritis menambahkan penalti ke *cost function* untuk mencegah *overfitting* dengan membatasi bobot model.

* **Chapter 5: Support Vector Machines (SVMs)**
    * [cite_start]**Ringkasan:** Detail mengenai SVM untuk klasifikasi dan regresi linear maupun non-linear, termasuk konsep seperti *soft margin*, *kernel*, dan matematika di baliknya.
    * [cite_start]**Teori Singkat:** Teori fundamental SVM adalah *large margin classification*: tujuannya adalah untuk menemukan "jalan" terluas yang memisahkan dua kelas, di mana batasnya ditentukan oleh *support vector*. [cite_start]Untuk data non-linear, SVM menggunakan *kernel trick*, sebuah teknik matematis yang memungkinkan model menemukan batas keputusan non-linear dengan secara implisit memetakan data ke ruang dimensi yang lebih tinggi tanpa benar-benar menambahkan fitur baru.

* **Chapter 6: Decision Trees**
    * [cite_start]**Ringkasan:** Meliputi cara melatih, memvisualisasikan, dan membuat prediksi dengan Decision Tree, termasuk algoritma CART dan teknik regularisasi.
    * [cite_start]**Teori Singkat:** Teori di balik *Decision Tree* adalah mempartisi data secara rekursif berdasarkan fitur dan ambang batas untuk menghasilkan *subset* yang paling murni. [cite_start]Algoritma CART (*Classification and Regression Tree*) melakukan ini dengan mencari pasangan fitur/ambang batas yang meminimalkan *cost function* (biasanya Gini *impurity* atau *entropy*) di setiap simpul.

* **Chapter 7: Ensemble Learning and Random Forests**
    * [cite_start]**Ringkasan:** Membahas metode untuk menggabungkan beberapa model (*ensemble*) untuk mencapai performa yang lebih baik, termasuk *bagging*, *pasting*, *boosting*, *stacking*, dan pembahasan mendalam tentang *Random Forest*.
    * [cite_start]**Teori Singkat:** Teori dasarnya adalah "kebijaksanaan orang banyak" (*wisdom of the crowd*), yang menyatakan bahwa menggabungkan prediksi dari sekelompok prediktor yang beragam seringkali menghasilkan prediksi yang lebih baik daripada prediktor tunggal terbaik[cite: 236]. [cite_start]Ini paling efektif ketika prediktor-prediktor tersebut bersifat independen[cite: 238]. [cite_start]*Bagging* dan *pasting* mengurangi varians [cite: 240][cite_start], sementara *boosting* mengurangi bias dengan melatih prediktor secara sekuensial.

* **Chapter 8: Dimensionality Reduction**
    * [cite_start]**Ringkasan:** Mengatasi "kutukan dimensionalitas" (*curse of dimensionality*) dengan teknik-teknik seperti PCA, Kernel PCA, dan LLE untuk mengurangi jumlah fitur dalam dataset.
    * [cite_start]**Teori Singkat:** Masalah teoritisnya adalah *curse of dimensionality*, di mana ruang berdimensi tinggi membuat data menjadi sangat jarang (*sparse*), sehingga sulit untuk menemukan pola. [cite_start]Dua pendekatan teoritis utama untuk mengatasinya adalah *proyeksi* (memproyeksikan data ke subruang berdimensi lebih rendah) dan *Manifold Learning*, yang mengasumsikan data terletak pada *manifold* berdimensi lebih rendah yang dapat "dibuka" atau "direntangkan".

* **Chapter 9: Unsupervised Learning Techniques**
    * [cite_start]**Ringkasan:** Menjelajahi metode *unsupervised* seperti *clustering* (K-Means, DBSCAN), *Gaussian Mixture Models*, dan aplikasinya dalam deteksi anomali dan estimasi kepadatan.
    * [cite_start]**Teori Singkat:** Teori *clustering* adalah tentang mengelompokkan instance yang mirip tanpa menggunakan label. [cite_start]Untuk K-Means, teorinya adalah proses iteratif: menetapkan instance ke *centroid* terdekat, kemudian menghitung ulang posisi rata-rata *centroid*, dan mengulanginya hingga konvergen. [cite_start]Untuk *Gaussian Mixture Models*, teorinya bersifat probabilistik, dengan asumsi bahwa data dihasilkan dari campuran beberapa distribusi Gaussian.

### Part II: Neural Networks and Deep Learning

* **Chapter 10: Introduction to Artificial Neural Networks with Keras**
    * [cite_start]**Ringkasan:** Pengenalan praktis tentang JST (Jaringan Saraf Tiruan) dan membangun model menggunakan Keras API yang sederhana dan kuat.
    * [cite_start]**Teori Singkat:** Teori JST terinspirasi dari neuron biologis. [cite_start]Sebuah neuron buatan (seperti TLU) menghitung jumlah tertimbang dari inputnya dan menerapkan fungsi aktivasi. [cite_start]Teori kunci untuk melatih MLP adalah *backpropagation*, yang pada dasarnya adalah *Gradient Descent* yang menggunakan *reverse-mode autodiff* untuk menghitung gradien secara efisien.

* **Chapter 11: Training Deep Neural Networks**
    * [cite_start]**Ringkasan:** Teknik dan praktik terbaik untuk melatih *deep neural network*, mengatasi tantangan seperti *vanishing/exploding gradients*.
    * [cite_start]**Teori Singkat:** Masalah teoritis utamanya adalah *unstable gradients* (*vanishing* atau *exploding*)[cite: 379]. [cite_start]Solusinya termasuk inisialisasi bobot yang tepat (Glorot/He) untuk menjaga varians di setiap lapisan [cite: 380][cite_start], menggunakan fungsi aktivasi non-saturasi (seperti ReLU dan variannya) [cite: 382][cite_start], dan *Batch Normalization* untuk menormalkan kembali aktivasi di setiap lapisan.

* **Chapter 12: Custom Models and Training with TensorFlow**
    * [cite_start]**Ringkasan:** Menyelami API tingkat rendah TensorFlow untuk membuat *loss function*, *layer*, model, dan *training loop* kustom untuk fleksibilitas maksimum.
    * **Teori Singkat:** Teori intinya adalah model eksekusi TensorFlow. [cite_start]Ia menggunakan *tensor* seperti array NumPy tetapi dengan dukungan GPU[cite: 423]. [cite_start]Kuncinya adalah *autodiff* (*reverse-mode*) untuk perhitungan gradien otomatis [cite: 426][cite_start], dan mengompilasi fungsi Python menjadi *computation graph* statis untuk optimisasi dan portabilitas.

* **Chapter 13: Loading and Preprocessing Data with TensorFlow**
    * [cite_start]**Ringkasan:** Metode untuk membangun alur pemuatan data yang efisien dan dapat diskalakan menggunakan Data API, format TFRecord, dan *layer* prapemrosesan Keras.
    * [cite_start]**Teori Singkat:** Teorinya adalah membangun alur data yang efisien[cite: 460]. [cite_start]Konsep kuncinya adalah *prefetching* untuk menumpang-tindihkan persiapan data (CPU) dan pelatihan model (GPU) [cite: 468][cite_start], pemrosesan data paralel, dan penggunaan format biner yang efisien seperti *TFRecord* dengan *Protocol Buffers*.

* **Chapter 14: Deep Computer Vision Using Convolutional Neural Networks (CNNs)**
    * [cite_start]**Ringkasan:** Tinjauan komprehensif arsitektur CNN (LeNet-5, AlexNet, GoogLeNet, ResNet, dll.) untuk tugas-tugas seperti klasifikasi gambar, deteksi objek, dan segmentasi semantik.
    * [cite_start]**Teori Singkat:** Arsitektur CNN terinspirasi dari korteks visual, menggunakan *local receptive fields*. [cite_start]Ide teoritis utamanya adalah *parameter sharing* (sebuah filter tunggal diterapkan di seluruh gambar untuk mendeteksi fitur di mana saja)  [cite_start]dan *pooling* untuk menciptakan invarian spasial dan mengurangi komputasi.

* **Chapter 15: Processing Sequences Using RNNs and CNNs**
    * [cite_start]**Ringkasan:** Teknik untuk menangani data sekuensial menggunakan *Recurrent Neural Networks* (RNN), LSTM, GRU, dan bagaimana CNN (seperti WaveNet) juga dapat digunakan untuk urutan.
    * [cite_start]**Teori Singkat:** Teori RNN adalah memproses urutan dengan mempertahankan *hidden state* (memori) yang diteruskan dari satu langkah waktu ke langkah berikutnya. Hal ini memungkinkan output jaringan pada waktu *t* menjadi fungsi dari semua input sebelumnya. [cite_start]Tantangannya adalah memori jangka pendek yang terbatas, yang diatasi oleh sel seperti LSTM dan GRU yang menggunakan *gates* untuk mengontrol aliran informasi.

* **Chapter 16: Natural Language Processing with RNNs and Attention**
    * **Ringkasan:** Menerapkan RNN pada tugas-tugas NLP seperti pembuatan teks dan analisis sentimen. [cite_start]Memperkenalkan mekanisme *attention* dan arsitektur Transformer yang kuat.
    * [cite_start]**Teori Singkat:** Teori inti untuk penerjemahan adalah arsitektur *Encoder-Decoder*, di mana satu RNN meng-encode kalimat sumber menjadi vektor konteks dan RNN lain men-decode-nya menjadi kalimat target. [cite_start]Mekanisme *attention* meningkatkannya dengan memungkinkan *decoder* untuk secara selektif fokus pada bagian-bagian relevan dari kalimat input pada setiap langkah. [cite_start]Arsitektur *Transformer* menghilangkan RNN sepenuhnya dan hanya mengandalkan mekanisme *self-attention*.

* **Chapter 17: Representation Learning and Generative Learning Using Autoencoders and GANs**
    * [cite_start]**Ringkasan:** Mempelajari *unsupervised learning* dengan *autoencoder* untuk *representation learning* dan menjelajahi *Generative Adversarial Networks* (GAN) untuk membuat data baru.
    * [cite_start]**Teori Singkat:** *Autoencoder* didasarkan pada teori *representation learning*: mereka belajar untuk mengompres (encode) input ke ruang laten (*latent space*) berdimensi rendah dan kemudian merekonstruksinya (decode), memaksa jaringan untuk mempelajari representasi data yang efisien. [cite_start]GAN didasarkan pada teori permainan: sebuah *generator* dan *discriminator* bersaing dalam permainan *zero-sum*, saling mendorong untuk meningkat hingga mencapai *Nash equilibrium*, di mana *generator* menghasilkan data yang realistis.

* **Chapter 18: Reinforcement Learning**
    * [cite_start]**Ringkasan:** Pengenalan dunia *Reinforcement Learning* yang menarik, dari konsep dasar hingga membangun agen pemain game dengan *Deep Q-Networks* dan TF-Agents.
    * [cite_start]**Teori Singkat:** Teorinya adalah mempelajari sebuah *policy* (strategi) untuk memaksimalkan akumulasi imbalan (*reward*) dalam sebuah lingkungan. [cite_start]Ini sering dimodelkan sebagai *Markov Decision Process* (MDP). [cite_start]Algoritma utamanya adalah *Policy Gradients* (mengoptimalkan *policy* secara langsung)  [cite_start]dan *Q-Learning* (mempelajari nilai dari pasangan state-action, atau Q-Values, menggunakan persamaan Bellman).

* **Chapter 19: Training and Deploying TensorFlow Models at Scale**
    * [cite_start]**Ringkasan:** Panduan praktis untuk menerapkan model TensorFlow ke lingkungan produksi dan meningkatkan skala pelatihan di berbagai perangkat dan server.
    * **Teori Singkat:** Teori di sini lebih berkaitan dengan desain sistem. [cite_start]Konsep utamanya adalah memisahkan model dari aplikasi dengan membuat *layanan prediksi* [cite: 715][cite_start], menggunakan format standar seperti *SavedModel* untuk portabilitas [cite: 716][cite_start], dan meningkatkan skala pelatihan menggunakan *data parallelism* (mereplikasi model) atau *model parallelism* (membagi model).

**Teknologi yang Digunakan** :<br>
- Python 3.x<br>
- Jupyter Notebook<br>
- TensorFlow 2<br>
- Keras (tf.keras)<br>
- Scikit-Learn<br>
- NumPy<br>
- Pandas<br>
- Matplotlib<br>
<br>
