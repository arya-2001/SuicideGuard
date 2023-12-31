# Arya_ML_DL_project
Developing a Text Classification Model for Identifying Suicidal  Ideation using Natural Language Processing Techniques (Machine Learning and Deep Learning) 

Suicide has become a serious social health issue in the modern society. Suicidal intent is people’s thoughts about committing suicide. The suicide of a person is a tragedy that deeply affects families, communities, and countries. According to the standardized rate of suicides per number of inhabitants worldwide, in 2022 there has been approximately 903,450 suicides and 18,069,000 unconsummated suicides, affecting people of all ages, countries, races, beliefs, social status, economic status, sex, etc. Depression is a prevalent mental disorder that can affect productivity in daily activities and might lead to suicidal thoughts or attempts. That means depression and suicide are related to each other. So, the aim of this project is to detect suicidal intent through depression detection by creating ML and DL models. Two proposed systems are developed for detecting suicidal intent from textual data. Proposed System 1 employs ML and DL approaches with features such as Bag-of-Words (unigram), TF-IDF, and bigram representations. Four ML classifiers (SVM, NB, LR, and RF) and four DL classifiers (LSTM, Bi-LSTM, CNN, and BERT) have been utilized. The evaluation of Proposed System 1 achieves accuracies of 82%, 87%, 77%, 73%, 71%, and 94% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Proposed System 2 incorporates a hybrid approach, combining LSTM, SVM, and the VADER Lexicon. The evaluation of Proposed System 2 yields accuracies of 97.4%, 97.14%, 99.2%, 91.5%, 94.4%, and 93.7% for the Reddit, Life_Corpus, CEASE, SWMH, SDCNL, and SDD datasets, respectively. Comparing the proposed systems with existing works, it is found that they outperform previous research works for the Reddit, Life_Corpus, CEASE, and SWMH datasets, achieving accuracy scores of 97.4%, 97.14%, 99.2%, and 91.5% respectively. Although the proposed model achieves promising results for the SDCNL and SDD datasets with accuracy scores of 94.4% and 94%, further research is needed to surpass the results of previous works.

Keywords: Machine Learning, Deep Learning, Depression, Suicide, Depression Detection, Suicidal Ideation, VADER, Text Classification.

The proposed methodology of this research work is divided into two categories. One of this 
is for Suicidal Ideation using Machine Learning and Deep Learning classifiers and other one is by
using hybrid model.
7.1: Proposed System 1:
7.1.1: Architectural framework of the Proposed System:
In the given diagram the architecture of the Proposed System 1 is presented.
Fig 7.1: Proposed System 1 architecture using ML & DL approach [71]
In Fig 7.1, the architectural framework of the proposed suicidal ideation system has been presented which 
incorporates four ML classifiers viz. SVM, NB, RF, LR and four DL classifiers viz. CNN, LSTM, Bi-LSTM, 
BERT. To conduct experimentation using ML approach, the feature extracted include TF-IDF (unigrams & 
bigrams) and BoW (unigrams & bigrams). To conduct experimentation using DL approach, one-hot 
encoding technique is used prior to building the classification model. Further details about the text 
classification approach have been discussed in the subsequent sections.
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 57
7.1.2: Flowchart of ML part of Proposed System 1:
In the given diagram (Fig 7.2) the flowchart of the ML part of Proposed System 1 is 
presented.
Fig 7.2: Flowchart of ML part of Proposed System 1
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 58
7.1.3: Flowchart of DL part of Proposed System 1:
In the given diagram (Fig 7.3) the flowchart of the DL part of Proposed System 1 is
presented.
Fig 7.3: Flowchart of DL part of Proposed System 1
7.1.4: Workflow of Proposed System1:
The working of our Proposed System1 is explained via different phases which are discussed 
as followsStep 1. Data Preprocessing: Collect relevant data related to suicidal ideation, such as text or 
social media posts, interviews, or medical records. For this work, six datasets are collected from 
various sources. Next clean the data by removing irrelevant information, special characters, and 
potentially sensitive information. Normalize the text data by converting it to lowercase, removing 
stop words, and applying lemmatization techniques.
Step 2. Feature Extraction: Extract meaningful features from the preprocessed texts. Some 
common approaches include Bag-of-Words (BoW); which represents the text as a frequency count 
of individual words or n-grams, TF-IDF, which weighs the importance of words based on their 
frequency in a document and the entire corpus, Sentiment Analysis; which analyzes the emotional 
tone of the text using sentiment lexicons or ML models.
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 59
Step 3. Encoding: Convert the extracted features into numerical representations that Deep 
Learning algorithms can process. If needed, perform additional encoding steps like one-hot 
encoding or label encoding for categorical variables.
Step 4. Data Splitting: n this part the dataset was splitted into train and test sub-units. The 
train unit is for training the classifiers in model and the test unit is for testing purposes for the 
results.
(a) In this project the datasets were splitted in 80:20 ratio for training and testing accordingly.
(b)To obtain a normalized result, k-fold cross-validation is often employed.
Step 5. Model Selection and Training: Apply various Machine Learning models on the 
training data. The models such as SVM, LR, RF and NB are traditional ML algorithms,while CNN, 
LSTM, Bi-LSTM, and BERT are DL models commonly used for text classification. Train each model 
using the training data and evaluate their performance using the validation set. Tune 
hyperparameters for each model to optimize their performance.
Step 6. Evaluation: Assess the performance of each model based on evaluation metrics such 
as accuracy, precision, recall, and F1-score. Select the model with the best performance on the 
validation set.
Step 7. Testing and Deployment: Assess the final model's performance on the testing set to 
obtain a more accurate estimate of its generalization ability.
7.2: Proposed System 2:
7.2.1: Architecture of Proposed System 2:
In Fig 7.4, the architectural framework of the novel hybrid suicidal ideation system has been presented which 
incorporates LSTM, and SVM classification approaches and has been trained using sentiment scores 
generated via VADER lexicon.
Fig 7.4: Architecture of Proposed System 2
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 60
7.2.2: Algorithm of Proposed System 2:
In the given steps the proposed algorithm of the Proposed System 2 is presented.
Step 1: Import the required libraries: “pandas”, “numpy”, “matplotlib.pyplot”, “seaborn”, 
“nltk”, “tensorflow” and others.
Step 2: Mount the Google Drive to access the dataset.
Step 3: Read the dataset (CSV file) using “pd.read_csv()” and store it in a DataFrame (‘df’).
Step 4: Define a mapping for the class labels.
Step 5: Preprocess the text data:
(a) Download the stopwords using “nltk.download('stopwords')”.
(b) Initialize a lemmatization using “WordNetLemmatizer()”.
(c) Iterate over the text data in the DataFrame and perform the following steps:
(d) Remove non-alphabetic characters using regex.
(e) Convert the text to lowercase.
(f) Split the text into words.
(g) Remove stopwords and perform lemmatization on each word.
(h) Join the processed words to form a sentence and append it to the “corpus” list.
Step 6: Perform one-hot encoding on the processed corpus using “one_hot()” and store the 
encoded representations in the “onehot_repr” list.
Step 7: Pad the one-hot encoded sequences to ensure they have the same length using
“pad_sequences()”.
Step 8: Download the VADER sentiment intensity analyzer using 
“nltk.download('vader_lexicon')”.
Step 9: Create a deep learning model using TensorFlow and Keras:
(a) Define the model architecture with an Embedding, Dense and LSTM layer.
(b) Compile the model with binary cross-entropy loss and the Adam optimizer.
Step 10: Split the data into training and testing sets using “KFold” from 
“sklearn.model_selection”.
Step 11: Perform k-fold cross-validation on the training data:
(a) Fit the LSTM model on the training data.
(b) Make predictions using the trained LSTM model.
(c) Calculate sentiment scores for text in each fold using VADER sentiment analysis.
(d) Combine the LSTM predictions with sentiment scores.
(e) Train a LinearSVC classifier on the combined features.
(f) Calculate the accuracy score for the classifier.
(g) Append the accuracy score to the “accuracy_scores” list.
Step 12: Calculate the mean accuracy score from the accuracy_scores.
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 61
7.2.3: Workflow of Proposed System 2:
The working of our Proposed System 2 is explained via different phases which are 
discussed as follows1. Data Collection: Dataset has a huge role on depression detection or suicidal 
ideation. There are lots of datasets available in online sources. For the project, firstly the idea 
is to focus on the online dataset available. In future work the implementation of the manual 
handwritten texts can also be possible. From the help of the all the analysed research papers 
the datasets from Kaggle, GitHub, reddit, life corpus etc were collected. Finally, the five 
datasets were shortlisted for the project work. The datasets were named as– “500 Reddit 
Post”, “SDCNL”, “Suicide and Depression Detection”, “Life_Corpus” and “SWMH”.
2. Reset Index: Various datasets contain various types of indexes. From those the all 
indexes may not be used. To overcome these problems the reset index part is proposed where 
the extra indexes of the dataset were removed. After that the different labels were changed 
in the form of our required type. As the various datasets contain various types of indexes, 
the reset index part for each dataset is different. The various libraries were used for this part. 
For example- pandas, NumPy etc. After this process the data is renamed or presented as reestablished data.
3. Data Pre-processing: The data pre-processing unit is an addition of four different 
sub-units. The first part is “Data cleaning”, the process of fixing or removing incorrect, 
corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset. The first part 
is “Stopwords removal”, the process of removing the words that occur commonly across all 
the documents. Where “nltk.corpus.stopwords('english')” is used. The second part is 
“Tokenization”, the process of replacing sensitive data with unique identification symbols 
that retain all the essential information about the data without compromising its security. 
Where the whole dataset is splitted then turned into tokens and finally arranged accordingly. 
The third and final part of data pre-processing is “Lemmatization'', the process of grouping 
together different forms of the same word. For example, Lemmatization of the words 'good', 
’better’ and ‘best’ would return 'good'. For the process the “WordNetLemmatizer()” is used.
4. Encoding: After data preprocessing unit the preprocessed data goes for encoding. 
encoding is the process of putting a sequence of characters (letters, numbers, punctuation, 
and certain symbols) into a specialized format. In this part “onehot encoding” is 
implemented. For this the “one_hot” is imported. One hot encoding is a technique used to 
represent categorical variables as numerical values in a Deep Learning model.
5. VADER Lexicon: The VADER (Valence Aware Dictionary and sEntiment Reasoner) 
lexicon is imported from NLTK.
6. Model Building: In this part the backbone of the LSTM classifier is constructed. In 
this section all layers of the LSTM model are created. In the end of this model the models for 
each classifier were compiled.
7. Data Splitting: k-fold cross-validation process by splitting the data into training 
and testing sets for each fold. The training and testing data are then used for further analysis, 
Developing a Text Classification Model for Identifying Suicidal Ideation using Natural Language Processing Techniques 62
such as training the model on the training data and evaluating its performance on the testing 
data.
8. Model Evaluation: In this scenario, an LSTM model is trained using the training 
data, X_train, and their corresponding labels, Y_train. The model undergoes 10 epochs of 
training, with a batch size of 64. Subsequently, the trained LSTM model is employed to make 
predictions on the testing data, X_test, using the predict () method. To gauge the sentiment 
of the text in the test set, sentiment scores are computed using the polarity_scores() method 
from the SentimentIntensityAnalyzer (SID) object, referred to as sid. These scores represent 
the overall sentiment polarity of each text, capturing its positive or negative connotation. To 
create a comprehensive feature matrix, the LSTM predictions (lstm_predictions) and the 
sentiment scores (test_sentiment_scores) are combined using np.column_stack() from 
NumPy, which horizontally stacks the arrays. The resulting combined_features array 
contains both sets of features. Next, an SVM model is trained using the combined feature 
matrix (combined_features) and the corresponding labels (Y_test). The SVM model learns to 
classify instances based on the combined features. Using the trained SVM model, predictions 
for the labels of the combined feature matrix are generated. Finally, the accuracy of the SVM 
predictions is calculated by comparing them to the true labels, providing a measure of the 
model's performance.

