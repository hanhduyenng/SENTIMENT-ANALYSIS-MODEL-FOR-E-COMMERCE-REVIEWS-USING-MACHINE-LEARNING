# SENTIMENT-ANALYSIS-MODEL-FOR-E-COMMERCE-REVIEWS-USING-MACHINE-LEARNING

Sentiment Analysis Model for E-Commerce Reviews is a machine learning project that aims to analyze and categorize customer reviews of e-commerce products to determine their sentiment—positive, negative, or neutral. Leveraging logistic regression, a widely used classification algorithm, this project provides valuable insights into understanding customer feedback and sentiment patterns.

## Table of Contents
- [Features](#features)
- [Objectives](#objectives)
- [Installation](#installation)
- [Documentation](#documentation)
- [Conclusion](#conclusion)

## Features

- Sentiment Classification: The model classifies reviews into three categories: positive, negative, or neutral, helping businesses gauge customer satisfaction.

- Scalability: Easily adaptable to process and analyze a large volume of e-commerce reviews, making it suitable for various industries.

- Accuracy: Logistic regression, known for its simplicity and efficiency, ensures accurate sentiment predictions.

- Customization: The project is designed for easy integration and customization into various e-commerce platforms and applications.
## Objectives
The project processes text data from customer reviews, applies machine learning techniques (Logistics Regression, Decision Tree, Naive Bayes) to analyze the customers sentiment based on a musical instrument review from Amazon, and generates insights that can drive business decisions. By learning from historical reviews, the model becomes more accurate over time.Understanding customer sentiment is crucial for e-commerce businesses. This project aims to:

- Identify product strengths and weaknesses through sentiment analysis.

- Improve customer satisfaction and product quality based on feedback.

- Make data-driven decisions for marketing and product development.

- Enhance user experiences by tailoring services to customer needs.
  
## Installation
To run this project, you'll need to install the following libraries. You can install them using `pip`.
```bash
# Basic libraries
pip install pandas
pip install numpy

# NLTK libraries
pip install nltk
pip install wordcloud
pip install nltk
nltk.download('stopwords')

# Machine Learning libraries
pip install scikit-learn

# Metrics libraries
pip install scikit-learn

# Visualization libraries
pip install matplotlib
pip install seaborn
pip install textblob
pip install plotly

# Other miscellaneous libraries
pip install imbalanced-learn

```


## Documentation
This project is centered on the essential stages of data handling, including data collection, data wrangling, cleaning, and preprocessing. We delve into the intricacies of the dataset, splitting it into training and testing sets to facilitate model evaluation.
- In order to refine the text data, it can be seen that TF-IDF Vectorizer offers the advantage of controlling term frequencies across documents and datasets. It penalizes frequently occurring words.
- Our dataset exhibits an imbalance, with a surplus of positive sentiments compared to negative and neutral sentiments. To address this imbalance, we employ the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE aims to rectify the class distribution by generating synthetic instances of the minority classOur dataset exhibits an imbalance, with a surplus of positive sentiments compared to negative and neutral sentiments. To address this imbalance, we employ the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE aims to rectify the class distribution by generating synthetic instances of the minority class.
- After the oversampling process, the data is reconstructed and several classification models can be applied for the processed data.
For the model selection, we have many at our disposal but will be utilizing the following: Logistics Regression, Decision Tree, Naive Bayes.

## **Conclusion**
- All of our models performed better than the baseline accuracy metric of ~80%, and the optimal models were LogisticRegression. These were determined not only in terms of overall raw accuracy, but in terms of variance and goodness of fit. Implementing hyperparameter tuning methods results in a significant accuracy improvement, raising the score from 88% to the desired 95%.
- Neutral reviews in our dataset often provide constructive criticism from buyers, offering valuable feedback that can be shared with sellers to enhance their products.

- Balancing the dataset led to significantly improved accuracy. Without balancing, precision may be high, but recall and consequently the F1 score would be negatively impacted. Thus, balancing the target feature is crucial for reliable results.

