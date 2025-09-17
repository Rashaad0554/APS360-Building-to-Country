# CNN-Based Country Recognition from Photographic Building Data

## 📌 Project Overview
This project explores the use of **deep learning for country-level geolocation of buildings and landmarks** from photographic data. By leveraging both a **baseline k-Nearest Neighbour (k-NN)** approach with handcrafted features and a **Convolutional Neural Network (CNN)** based on **EfficientNet-B5 transfer learning**, we aimed to classify buildings into one of five countries with strong architectural diversity.

The project was developed as part of **APS360: Applied Fundamentals of Machine Learning** at the University of Toronto.

---

## ✨ Features
- **Custom Dataset**: Scraped ~3,500 images (700 per country) using Bing via iCrawler, followed by CLIP-based filtering to remove irrelevant images.  
- **Baseline Model**: k-NN with Color Histogram and SIFT descriptors, achieving **71.41% accuracy**.  
- **Primary Model**: EfficientNet-B5 with transfer learning from ImageNet, achieving **77.45% test accuracy**.  
- **Robust Preprocessing**: Balanced dataset, image resizing to 260×260, and train/validation/test split (70/15/15).  
- **Overfitting Mitigation**: Label smoothing, MixUp augmentation, and Adam optimizer with weight decay.  

---

## 🧑‍💻 Methodology
1. **Data Collection & Filtering**  
   - Scraped using iCrawler with country-specific prompts.  
   - Filtered using CLIP similarity scores to remove non-building and low-quality images.  

2. **Baseline Model**  
   - Feature extraction with **Color Histograms** and **SIFT descriptors**.  
   - Classification using **k-NN (k=1 best result)**.  

3. **Primary Model**  
   - Transfer learning with **EfficientNet-B5** pretrained on ImageNet.  
   - Global average pooling + fully connected softmax output layer.  
   - Training with adaptive learning rate scheduling and regularization.  

4. **Evaluation**  
   - Baseline achieved **71.41% accuracy**.  
   - EfficientNet achieved **77.45% accuracy** with improved generalization.  
   - Model performance was higher on unique landmarks but struggled with modern/similar architectural styles.  

---

## 📊 Results
- **Baseline (k-NN + Color Histogram + SIFT)**: 71.41%  
- **EfficientNet-B5 (Transfer Learning)**: 77.45%  

Example predictions:
- Louvre (France): Predicted France (37.8%) vs Italy (29.1%).  
- Mexico modern building: Misclassified as Greece due to architectural similarity.  

---

## ⚖️ Ethical Considerations
- Restricted to **country-level classification** to avoid privacy concerns.  
- Data sourced from **publicly available platforms** for academic use.  
- Acknowledges potential misuse in surveillance or location inference.  

---

## 🚀 Future Work
- Expand dataset to more countries for broader coverage.  
- Explore advanced geolocation architectures such as **PlaNet, StreetCLIP, and PIGEON**.  
- Implement fine-grained city-level geolocation while mitigating privacy risks.  
- Reduce bias from overrepresented landmarks and modern architectural similarities.  

---

## 🛠️ Tech Stack
- **Python**, **PyTorch**, **OpenCV**  
- **EfficientNet-B5 (ImageNet pretrained)**  
- **iCrawler** for data scraping  
- **CLIP** for dataset filtering  

---

## 👥 Authors
- Harish Babu  
- Rashaad Anwar  
- Michael Zheng  

---

## 📖 References
- Weyand et al., *PlaNet: Photo Geolocation with CNNs* (2016)  
- Haas et al., *StreetCLIP* (2023), *PIGEON* (2024)  
- Hays & Efros, *IM2GPS* (2008)  
- Al-Habashna, *Building Type Classification with CNNs* (2022)  
- Roboflow, *EfficientNet Guide* (2024)  
