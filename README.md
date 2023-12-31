# Arya_ML_DL_project

## Developing a Text Classification Model for Identifying Suicidal  Ideation using Natural Language Processing Techniques (Machine Learning and Deep Learning) 

Suicide has become a serious social health issue in the modern society. Suicidal intent is people’s thoughts about committing suicide. The suicide of a person is a tragedy that deeply affects families, communities, and countries. According to the standardized rate of suicides per number of inhabitants worldwide, in 2022 there has been approximately 903,450 suicides and 18,069,000 unconsummated suicides, affecting people of all ages, countries, races, beliefs, social status, economic status, sex, etc. Depression is a prevalent mental disorder that can affect productivity in daily activities and might lead to suicidal thoughts or attempts. That means depression and suicide are related to each other. So, the aim of this project is to detect suicidal intent through depression detection by creating ML and DL models. Two proposed systems are developed for detecting suicidal intent from textual data. Proposed System 1 employs ML and DL approaches with features such as Bag-of-Words (unigram), TF-IDF, and bigram representations. Four ML classifiers (SVM, NB, LR, and RF) and four DL classifiers (LSTM, Bi-LSTM, CNN, and BERT) have been utilized. The evaluation of Proposed System 1 achieves accuracies of 82%, 87%, 77%, 73%, 71%, and 94% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Proposed System 2 incorporates a hybrid approach, combining LSTM, SVM, and the VADER Lexicon. The evaluation of Proposed System 2 yields accuracies of 97.4%, 97.14%, 99.2%, 91.5%, 94.4%, and 93.7% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Comparing the proposed systems with existing works, it is found that they outperform previous research works for the Reddit, Life_Corpus, CEASE, and SWMH datasets, achieving accuracy scores of 97.4%, 97.14%, 99.2%, and 91.5% respectively. Although the proposed model achieves promising results for the SDCNL and SDD datasets with accuracy scores of 94.4% and 94%, further research is needed to surpass the results of previous works.

### Keywords: Machine Learning, Deep Learning, Depression, Suicide, Depression Detection, Suicidal Ideation, VADER, Text Classification.

# Proposed Methodology

## System 1: Suicidal Ideation using ML & DL Classifiers

### Architectural Framework
![ML   DL architecture](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/20c21ace-ebd5-47ed-845e-0b269ccf0047)

The proposed system employs ***four ML classifiers (SVM, NB, RF, LR)*** and ***four DL classifiers (CNN, LSTM, Bi-LSTM, BERT)*** for identifying suicidal ideation. Feature extraction includes **TF-IDF (unigrams & bigrams)** and **BoW (unigrams & bigrams)** for ML, and **one-hot encoding** for DL.

#### ML Part Flowchart
![Machine Learning flowchart](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/17fe5e37-55a8-4cac-afd3-0a8408bef91f)

#### DL Part Flowchart
![deep learning flowchart](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/d06ae2db-fc85-48a2-83c4-0ba45a33cba4)

### Workflow

1. **Data Preprocessing:**
   - Collect relevant data related to suicidal ideation.
   - Clean and normalize data by removing irrelevant information, special characters, and applying lemmatization.

2. **Feature Extraction:**
   - Extract meaningful features using approaches like **BoW**, **TF-IDF**, and **Sentiment Analysis**.

3. **Encoding:**
   - Convert features into numerical representations suitable for DL algorithms.

4. **Data Splitting:**
   - Split the dataset into train and test units (80:20 ratio).
   - Use k-fold cross-validation for normalized results.

5. **Model Selection and Training:**
   - Apply ML models (**SVM, LR, RF, NB**) and DL models (**CNN, LSTM, Bi-LSTM, BERT**).
   - Optimize hyperparameters for each model.

6. **Evaluation:**
   - Assess model performance using metrics such as accuracy, precision, recall, and F1-score.

7. **Testing and Deployment:**
   - Evaluate the final model on the testing set for generalization.

## System 2: Hybrid Model

### Architecture
![Novel Hybrid system archicecture](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/0306b6b1-2692-4f7d-bceb-12d51c16911a)

### Algorithm & Implementation

#### System 2: Hybrid Model

##### Algorithm

1. **Import Libraries:**
   - Import the required libraries: "pandas", "numpy", "matplotlib.pyplot", "seaborn", "nltk", "tensorflow", and others.

2. **Mount Google Drive:**
   - Mount the Google Drive to access the dataset.

3. **Read Dataset:**
   - Read the dataset (CSV file) using "pd.read_csv()" and store it in a DataFrame ('df').

4. **Define Class Labels Mapping:**
   - Define a mapping for the class labels.

5. **Preprocess Text Data:**
   - Download stopwords using "nltk.download('stopwords')".
   - Initialize lemmatization using "WordNetLemmatizer()".
   - Iterate over the text data in the DataFrame and perform the following steps:
     - Remove non-alphabetic characters using regex.
     - Convert the text to lowercase.
     - Split the text into words.
     - Remove stopwords and perform lemmatization on each word.
     - Join the processed words to form a sentence and append it to the "corpus" list.

6. **Perform One-Hot Encoding:**
   - Perform one-hot encoding on the processed corpus using "one_hot()" and store the encoded representations in the "onehot_repr" list.

7. **Pad Sequences:**
   - Pad the one-hot encoded sequences to ensure they have the same length using "pad_sequences()".

8. **Download VADER Lexicon:**
   - Download the VADER sentiment intensity analyzer using "nltk.download('vader_lexicon')".

9. **Create Deep Learning Model:**
   - Define the model architecture with an Embedding, Dense, and LSTM layer.
   - Compile the model with binary cross-entropy loss and the Adam optimizer.

10. **Split Data into Training and Testing Sets:**
   - Split the data into training and testing sets using "KFold" from "sklearn.model_selection".

11. **Perform K-Fold Cross-Validation on the Training Data:**
    - Fit the LSTM model on the training data.
    - Make predictions using the trained LSTM model.
    - Calculate sentiment scores for text in each fold using VADER sentiment analysis.
    - Combine the LSTM predictions with sentiment scores.
    - Train a LinearSVC classifier on the combined features.
    - Calculate the accuracy score for the classifier.
    - Append the accuracy score to the "accuracy_scores" list.

12. **Calculate Mean Accuracy Score:**
    - Calculate the mean accuracy score from the accuracy_scores.

See code implementation in the `algorithm_system_2.py` file.

### Workflow

1. **Data Collection:**
   - Collect datasets from various sources (e.g., Kaggle, GitHub, Reddit, life corpus).

2. **Reset Index:**
   - Remove unnecessary indexes and standardize labels.

3. **Data Pre-processing:**
   - Clean data, remove stopwords, tokenize, and lemmatize.

4. **Encoding:**
   - Utilize **one-hot encoding** for numerical representation.

5. **VADER Lexicon:**
   - Import VADER lexicon for sentiment analysis.

6. **Model Building:**
   - Construct the backbone of the LSTM classifier.

7. **Data Splitting:**
   - Implement k-fold cross-validation.

8. **Model Evaluation:**
   - Train LSTM model, make predictions, and calculate sentiment scores.
   - Combine LSTM predictions with sentiment scores.
   - Train SVM classifier on combined features and evaluate accuracy.


