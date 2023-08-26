### Kaggle Notebook Link: https://www.kaggle.com/code/shirshmall/cotton-disease-prediction
### Data Set Link: https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset

## Project Title: Cotton Disease Prediction - Image Classification (Multi-class classification)

#### Project Overview
This project aims to develop an image classification model for predicting diseases in cotton plants. The dataset, sourced from Kaggle, contains images of diseased and non-diseased cotton plants and leaves, captured using a mobile phone camera. The primary objective is to accurately categorize these images into distinct disease classes.

#### Model Preparation
**Step 1: Theoretical Foundation and CNN Model Creation**
- Gain theoretical knowledge of Convolutional Neural Networks (CNNs).
- Start by building a CNN model with basic layers to establish a baseline.
- Experiment with various hyperparameters like strides, padding, and architecture depth.
- Estimate the optimal number of convolutional layers and feature maps required.

**Step 1 Continued: Addressing Overfitting**
- Encounter the challenge of overfitting, where the model performs well on training data but poorly on unseen data.
- Implement techniques like Dropout layers to mitigate overfitting.
- Determine the appropriate dropout percentage to balance regularization without sacrificing model performance.
- Explore L1 and L2 Regularization along with Batch Normalization to further combat overfitting.

**Step 1 Continued: Data Augmentation**
- Introduce Data Augmentation techniques to artificially increase the diversity of the training dataset.
- Techniques include random flips, rotations, and zooming of images.
- Data Augmentation helps prevent overfitting by providing the model with more varied examples.

**Step 1 Results and Challenges**
- Despite efforts, the model's accuracy remains at 0.74 due to persistent overfitting.

**Step 2: Hyperparameter Tuning with Keras Tuner**
- Utilize Keras Tuner to search for optimal hyperparameters and architecture configurations.
- Keras Tuner employs techniques like Random Search or Hyperband to efficiently explore the hyperparameter space.

**Step 3: Transfer Learning with Established CNN Architectures**
- Implement Transfer Learning using established CNN models: ResNet50, MobileNetV2, InceptionV3, and VGG16.
- Leverage pre-trained weights learned from large datasets to boost model performance.
- ResNet50 emerges as the top performer in terms of validation accuracy.

**Step 3 Continued: VGG16 Training Time**
- Note that while VGG16 shows promising training accuracy, its training time is significantly longer (around 10 minutes per epoch).

**Step 3 Continued: Fine-tuning ResNet50**
- Choose ResNet50 as the final model and perform fine-tuning.
- Fine-tuning involves adjusting certain layers of the pre-trained model to better adapt it to the specific task.

**Step 3 Results**
- Achieve an impressive 97% accuracy on the validation dataset with the fine-tuned ResNet50 model.

#### Model Deployment
- Develop a user-friendly web application using Streamlit.
- Streamlit enables easy creation of interactive web apps for showcasing machine learning models.
- Users can upload images through the app to receive predictions about the health status of cotton plants.

#### Conclusion
This project exemplifies the journey from data collection to model selection and deployment. Techniques such as data augmentation, hyperparameter tuning, and transfer learning contribute to building a robust solution for real-world cotton disease prediction.


### Web app screenshots
![dis leaf](https://github.com/shirsh10mall/Cotton-Disease-Prediction-Image-Classification-/assets/87264071/75742b5f-6b72-4655-9f8d-ba2cbb5fd298)
<br>
![dis plant](https://github.com/shirsh10mall/Cotton-Disease-Prediction-Image-Classification-/assets/87264071/66e02509-cceb-4bef-b47e-3a9b1277669a)
<br>
![freash leaf](https://github.com/shirsh10mall/Cotton-Disease-Prediction-Image-Classification-/assets/87264071/2d4bc6de-b84f-4bee-a91e-d8a0cb16828d)
<br>
![fresh plant](https://github.com/shirsh10mall/Cotton-Disease-Prediction-Image-Classification-/assets/87264071/108882f9-7a0b-404e-8259-21968dc29110)
<br>

---

**Sources:**
- [Convolutional Neural Networks (CNNs): An Illustrated Explanation](https://towardsdatascience.com/convolutional-neural-networks-cnns-an-illustrated-explanation-8d29b0aa23ec)
- [A Comprehensive Introduction to Different Types of Regularization Techniques](https://www.analyticsvidhya.com/blog/2021/08/a-comprehensive-introduction-to-different-types-of-regularization-techniques-in-deep-learning/)
- [Data Augmentation in Deep Learning: An Overview](https://neptune.ai/blog/data-augmentation-in-deep-learning-an-overview)
- [Keras Tuner: Hyperparameter Tuning for Humans](https://keras.io/keras_tuner/)
- [Transfer Learning and Fine-Tuning Explained](https://neptune.ai/blog/transfer-learning-and-fine-tuning)
- [Streamlit: The Fastest Way to Build Data Apps](https://www.streamlit.io/)

