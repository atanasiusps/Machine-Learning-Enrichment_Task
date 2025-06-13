# **Rangkuman Repository**

Repositori ini berisi solusi dan implementasi latihan-latihan dalam buku karya Aurélien Géron, *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd edition.*

## **Tujuan**

Tujuan dari repositori ini adalah untuk memenuhi nilai mata kuliah Machine Learning. Repositori ini mencakup implementasi kode untuk latihan yang ditemukan di akhir setiap bab, serta kode dari dalam bab-bab pada buku.

---

### Part I: The Fundamentals of Machine Learning

#### **Chapter 1: The Machine Learning Landscape**
- **Ringkasan:** Pengenalan konsep-konsep inti Machine Learning, kategori utamanya (supervised, unsupervised, dll.), dan tantangan utama seperti overfitting dan underfitting.
- **Teori Singkat:** Teori dasarnya adalah bahwa sebuah sistem dapat belajar dari pengalaman (data). Hal ini didefinisikan secara formal oleh Tom Mitchell: sebuah program komputer dikatakan belajar dari pengalaman E sehubungan dengan tugas T dan ukuran kinerja P, jika kinerjanya pada T, yang diukur dengan P, meningkat seiring dengan pengalaman E.

#### **Chapter 2: End-to-End Machine Learning Project**
- **Ringkasan:** Panduan praktis melalui sebuah proyek ML lengkap, mulai dari membingkai masalah dan mendapatkan data hingga fine-tuning dan meluncurkan model.
- **Teori Singkat:** Metodologi terstruktur untuk proyek ML. Ini melibatkan pemilihan ukuran kinerja yang tepat seperti *Root Mean Square Error* (RMSE) untuk mengevaluasi dan mengoptimalkan model secara objektif.

#### **Chapter 3: Classification**
- **Ringkasan:** Pembahasan mendalam tentang tugas klasifikasi, metrik performa, dan algoritma umum menggunakan dataset MNIST sebagai studi kasus.
- **Teori Singkat:** Teori evaluasi klasifikasi berpusat pada metrik seperti *confusion matrix*, *precision*, *recall*, dan kurva ROC untuk analisis performa.

#### **Chapter 4: Training Models**
- **Ringkasan:** Regresi Linier, *Gradient Descent*, Regresi Polinomial, dan model regularisasi seperti *Ridge*, *Lasso*, dan *Elastic Net*.
- **Teori Singkat:** Intinya adalah minimisasi *cost function* dengan metode analitik (*Normal Equation*) atau iteratif (*Gradient Descent*), dan regularisasi untuk menghindari overfitting.

#### **Chapter 5: Support Vector Machines (SVMs)**
- **Ringkasan:** SVM untuk klasifikasi dan regresi linear maupun non-linear.
- **Teori Singkat:** *Large margin classification*, *support vectors*, dan *kernel trick* untuk klasifikasi non-linear.

#### **Chapter 6: Decision Trees**
- **Ringkasan:** Pelatihan dan visualisasi Decision Tree dengan algoritma CART dan teknik regularisasi.
- **Teori Singkat:** Pemisahan data rekursif berdasarkan fitur/threshold untuk meminimalkan impurity (Gini atau entropy).

#### **Chapter 7: Ensemble Learning and Random Forests**
- **Ringkasan:** Metode *bagging*, *pasting*, *boosting*, *stacking*, dan *Random Forests*.
- **Teori Singkat:** "Wisdom of the crowd" — gabungan model yang saling independen biasanya lebih baik dari satu model tunggal.

#### **Chapter 8: Dimensionality Reduction**
- **Ringkasan:** Teknik seperti PCA, Kernel PCA, dan LLE untuk reduksi fitur.
- **Teori Singkat:** *Curse of dimensionality* diatasi dengan proyeksi dan *manifold learning*.

#### **Chapter 9: Unsupervised Learning Techniques**
- **Ringkasan:** Clustering (K-Means, DBSCAN), Gaussian Mixture Models, dan aplikasinya.
- **Teori Singkat:** Pengelompokan tanpa label. K-Means menggunakan iterasi centroid, GMM berbasis distribusi probabilistik Gaussian.

---

### Part II: Neural Networks and Deep Learning

#### **Chapter 10: Introduction to Artificial Neural Networks with Keras**
- **Ringkasan:** Dasar-dasar JST dan penggunaan Keras.
- **Teori Singkat:** Neuron buatan, fungsi aktivasi, dan *backpropagation* untuk pelatihan.

#### **Chapter 11: Training Deep Neural Networks**
- **Ringkasan:** Praktik terbaik pelatihan DNN, mengatasi *vanishing/exploding gradients*.
- **Teori Singkat:** Bobot awal (Glorot/He), ReLU, *Batch Normalization* untuk menjaga stabilitas pelatihan.

#### **Chapter 12: Custom Models and Training with TensorFlow**
- **Ringkasan:** API tingkat rendah untuk membuat *loss*, *layer*, dan *training loop* kustom.
- **Teori Singkat:** Tensor, *autodiff*, dan *computation graph* untuk efisiensi dan portabilitas.

#### **Chapter 13: Loading and Preprocessing Data with TensorFlow**
- **Ringkasan:** Data pipeline dengan `tf.data`, TFRecord, dan *preprocessing layers*.
- **Teori Singkat:** *Prefetching*, *parallel processing*, dan format biner efisien.

#### **Chapter 14: Deep Computer Vision Using CNNs**
- **Ringkasan:** Arsitektur CNN seperti LeNet-5, AlexNet, ResNet, dll.
- **Teori Singkat:** *Local receptive fields*, *parameter sharing*, dan *pooling* untuk ekstraksi fitur.

#### **Chapter 15: Processing Sequences Using RNNs and CNNs**
- **Ringkasan:** Pemrosesan urutan menggunakan RNN, LSTM, GRU, dan CNN.
- **Teori Singkat:** *Hidden state* untuk menangkap konteks urutan, serta penggunaan *gates* di LSTM/GRU.

#### **Chapter 16: NLP with RNNs and Attention**
- **Ringkasan:** Penerapan RNN di NLP dan pengenalan *attention* dan *Transformer*.
- **Teori Singkat:** *Encoder-Decoder*, *attention mechanism*, dan *self-attention* di Transformer.

#### **Chapter 17: Representation Learning and Generative Learning**
- **Ringkasan:** *Autoencoder* dan *GANs*.
- **Teori Singkat:** *Autoencoder* untuk kompresi laten; *GAN* menggunakan persaingan antara *generator* dan *discriminator*.

#### **Chapter 18: Reinforcement Learning**
- **Ringkasan:** Pengantar RL dengan DQN dan TF-Agents.
- **Teori Singkat:** *Policy* dan *Q-Learning*, serta konsep MDP dan *Bellman equation*.

#### **Chapter 19: Training and Deploying TensorFlow Models at Scale**
- **Ringkasan:** Penerapan model ke produksi dan skala pelatihan.
- **Teori Singkat:** *SavedModel*, *prediction service*, *data/model parallelism*.

---

## **Teknologi yang Digunakan**

- Python 3.x  
- Jupyter Notebook  
- TensorFlow 2  
- Keras (tf.keras)  
- Scikit-Learn  
- NumPy  
- Pandas  
- Matplotlib

