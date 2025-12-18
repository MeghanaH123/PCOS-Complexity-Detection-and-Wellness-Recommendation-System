# PCOS-Complexity-Detection-and-Wellness-Recommendation-System

******* Overview *******

Polycystic Ovary Syndrome (PCOS) is a complex hormonal disorder that often goes undetected due to vague symptoms and fragmented diagnostic approaches.
This project presents a hybrid AI-based system that integrates clinical data analysis and ultrasound image processing to detect the presence of PCOS, classify its complexity (stage), and generate personalized wellness recommendations.

The system combines ensemble machine learning, GAN-based data augmentation, and deep learning–based medical image segmentation, delivered through a Flask-based web application.

****** Objectives ******

Detect PCOS using clinical parameters

Segment ovarian cysts from ultrasound images using U-Net

Classify PCOS complexity into Early Stage or Advanced Stage

Provide stage-specific diet, exercise, and lifestyle recommendations

Offer a simple, accessible web-based interface for users and clinicians

******* Key Features ******

-> Clinical PCOS Detection using Ensemble Learning

-> Voting Classifier combining:

Gradient Boosting

Logistic Regression

Random Forest

-> GAN-Based Data Augmentation to handle limited and imbalanced clinical datasets

-> PCA for Dimensionality Reduction during clinical model training

-> U-Net–Based Ultrasound Image Segmentation

-> Cyst Feature Extraction (count, area, average size, shape/circularity)

-> Image-Based PCOS Stage Classification

-> Personalized Wellness, Diet & Exercise Recommendations

-> Flask Web Application with HTML/CSS Frontend

******* Tech Stack *******
==> Machine Learning & Deep Learning

Python

Voting Classifier (Gradient Boosting, Logistic Regression, Random Forest)

Generative Adversarial Networks (GANs)

Principal Component Analysis (PCA)

U-Net (PyTorch)

Scikit-learn

NumPy, Pandas

OpenCV

******* Backend ******

Flask (Python)

Joblib / Pickle for model serialization

REST-style request handling

******* Frontend *******

HTML

CSS

JavaScript

Jinja2 Templates

****** Tools & Platforms *****

Jupyter Notebook

VS Code

Google Colab / Kaggle (model training)

************* System Architecture ************

The system follows a three-tier architecture:

Presentation Layer – HTML/CSS frontend for user interaction

Application Layer – Flask backend handling requests, predictions, and responses

Model Layer – ML and DL models for detection, segmentation, staging, and recommendation

************** System Workflow **************

Clinical data is augmented using GANs

PCA is applied to reduce dimensionality

A Voting Classifier predicts PCOS presence from clinical features

Ultrasound images are segmented using U-Net

Cyst features (count, area, circularity) are extracted

An image-based classifier determines PCOS stage (Early / Advanced)

A rule-based recommendation engine generates personalized wellness plans

Results are displayed via the web interface

*********** Key Contributions *****************

Unified clinical and ultrasound-based PCOS detection

Improved robustness using ensemble learning and GAN augmentation

Automated ovarian cyst detection using deep learning

End-to-end integration of AI models with a web application

Lightweight, low-cost system suitable for clinics and rural healthcare setups

***** Testing & Validation *******

Unit testing using unittest and pytest

Automation testing using Selenium

Edge-case handling for invalid inputs and low-quality images

Cross-browser compatibility verified

******** Future Enhancements ****************

Cloud deployment (AWS/GCP)

Explainable AI (SHAP / LIME)

Patient login and report history

Mobile-first UI

Integration with wearable health data

Multilingual support
