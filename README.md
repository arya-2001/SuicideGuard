# SuicideGuard 23.3.0

## Developing a Text Classification Model for Identifying Suicidal  Ideation using Natural Language Processing Techniques (Machine Learning and Deep Learning) 

Suicide has become a serious social health issue in the modern society. Suicidal intent is people’s thoughts about committing suicide. The suicide of a person is a tragedy that deeply affects families, communities, and countries. According to the standardized rate of suicides per number of inhabitants worldwide, in 2022 there has been approximately 903,450 suicides and 18,069,000 unconsummated suicides, affecting people of all ages, countries, races, beliefs, social status, economic status, sex, etc. Depression is a prevalent mental disorder that can affect productivity in daily activities and might lead to suicidal thoughts or attempts. That means depression and suicide are related to each other. So, the aim of this project is to detect suicidal intent through depression detection by creating ML and DL models. Two proposed systems are developed for detecting suicidal intent from textual data. Proposed System 1 employs ML and DL approaches with features such as Bag-of-Words (unigram), TF-IDF, and bigram representations. Four ML classifiers (SVM, NB, LR, and RF) and four DL classifiers (LSTM, Bi-LSTM, CNN, and BERT) have been utilized. The evaluation of Proposed System 1 achieves accuracies of 82%, 87%, 77%, 73%, 71%, and 94% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Proposed System 2 incorporates a hybrid approach, combining LSTM, SVM, and the VADER Lexicon. The evaluation of Proposed System 2 yields accuracies of 97.4%, 97.14%, 99.2%, 91.5%, 94.4%, and 93.7% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Comparing the proposed systems with existing works, it is found that they outperform previous research works for the Reddit, Life_Corpus, CEASE, and SWMH datasets, achieving accuracy scores of 97.4%, 97.14%, 99.2%, and 91.5% respectively. Although the proposed model achieves promising results for the SDCNL and SDD datasets with accuracy scores of 94.4% and 94%, further research is needed to surpass the results of previous works.

### Keywords: Machine Learning, Deep Learning, Depression, Suicide, Depression Detection, Suicidal Ideation, VADER, Text Classification.

# Proposed Methodology

## System 1: Suicidal Ideation using ML & DL Classifiers

### Architectural Framework :
![ML   DL architecture](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/20c21ace-ebd5-47ed-845e-0b269ccf0047)

The proposed system employs ***four ML classifiers (SVM, NB, RF, LR)*** and ***four DL classifiers (CNN, LSTM, Bi-LSTM, BERT)*** for identifying suicidal ideation. Feature extraction includes **TF-IDF (unigrams & bigrams)** and **BoW (unigrams & bigrams)** for ML, and **one-hot encoding** for DL.

#### ML Part Flowchart :
![Machine Learning flowchart](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/17fe5e37-55a8-4cac-afd3-0a8408bef91f)

#### DL Part Flowchart :
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

### Architecture :
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

See code implementation in the `Arya_ML_DL_project\Source Codes\Novel Hybrid System.ipynb` file.

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

     
### Total Result Table:
***Table 1: Best Accuracy of selected datasets using ML & DL approach***

![Screenshot 2023-12-31 225716](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/711166d1-bf39-4915-857e-fb8e9c94f6f2)

### Result of Proposed System 2:
***Table 2: Accuracy of selected datasets using Proposed System 2***

![Screenshot 2023-12-31 230837](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/2c897592-1b59-46cb-9a5b-98bc44bcfae3)

### Comparison Table:
***Table 3: Comparison of accuracy between existing research work and proposed work***

![image](https://github.com/arya-2001/Arya_ML_DL_project/assets/82049658/dcb3ce24-9e9e-4ad1-b0b2-2156af77eb5c)

## Conclusion

The primary goal of this project is to identify suicidal intent through depression detection in texts using a combination of Machine Learning (ML) and Deep Learning (DL) techniques. The approach involved a thorough analysis of previously conducted research works, identifying gaps, and selecting appropriate features, ML, and DL techniques. Six diverse datasets were collected from these research papers to facilitate experimentation.

### Proposed System 1

A comprehensive workflow was proposed for detecting suicidal intent from textual data using the ML and DL approach in Proposed System 1. The methodology included the design of a system architecture and a flowchart. Four features—BoW (unigram), TF-IDF, BoW (bigram), and TF-IDF (bigram)—were extracted, and individual and combined features were employed as inputs for four ML classifiers (SVM, NB, LR, and RF) and four DL classifiers (LSTM, Bi-LSTM, CNN, and BERT). The model achieved notable accuracy results for different datasets, with scores as follows: Reddit (82%), Life_Corpus (87%), CEASE (77%), SWMH (73%), SDCNL (71%), and SDD (94%).

### Proposed System 2

A separate workflow was proposed for detecting suicidal intent from textual data using a hybrid approach in Proposed System 2. This approach involved the construction of a system architecture and a flowchart. The model integrated LSTM, SVM, and VADER Lexicon. After evaluation, the model demonstrated high accuracy across datasets: Reddit (97.4%), Life_Corpus (97.14%), CEASE (99.2%), SWMH (91.5%), SDCNL (94.4%), and SDD (93.7%).

### Comparison and Implications

Comparing the proposed systems with existing works, the results indicate superior performance for four datasets: Reddit, Life_Corpus, CEASE, and SWMH, achieving accuracy scores of 97.4%, 97.14%, 99.2%, and 91.5%, respectively. While SDCNL and SDD also showed promising results with accuracy scores of 94.4% and 94%, they did not surpass the accuracy achieved by previous research on these datasets. This suggests that further research is needed to enhance the performance on these specific datasets.

In conclusion, the proposed systems demonstrate robust capabilities in detecting suicidal intent through text analysis, showcasing competitive accuracy scores across various datasets. The findings provide valuable insights and lay the groundwork for future research in this critical domain.



[# For ***Dataset*** and ***Output Demo*** Click on this Link](https://drive.google.com/drive/folders/1Ligplfr9LwaU0b27CJAJ0ZPv3LMbW-w7?usp=sharing)
