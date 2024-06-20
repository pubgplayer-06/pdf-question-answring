# pdf-question-answering
Our overall models architecture consists of a sentence selector and a QA model. 
The sentence selector computes a selection score for each sentence and we will set a threshold value above which only sentences should be taken.
I had diferent sentence selector for distillbert qa model. You can go through each code in the code section.
basic:
 without using any sentence selector, directly trained the data on QA model
tfidf:
i tried to calculate tfidf embeddings of each sentence and find the similarity between them .
word2vec:
here we train the word2vec model on whole context and find word2vec embeddings for each sentnecs by calculating the mean of the embedding of each word in the sentence. 
lstm+predefined_embeddings:
Generally we train lstm models on data , according to which the weights get adjusted to minimize te loss . but here we are taking the embeddings directly . so i thought  of pre-loading the  embedding layers weights with the file "glove.6B.100d.txt" (available on kaggle). Now i had taken the embeddings of sentences using this model and calculated the similarity score.
lstm+embeddings_2:
just like the previous case but here i had increaed the layers of bidirectional lstms .
lstm+cnn:
CNNs excel at retrieving local and position-invariant characteristics, but RNNs excel at classification based on a long-range semantic dependency instead of local key-value pairs. It shows out that CNNs+ LSTMs perform admirably when used to NLP problems.So, I had used embeddings of this model to claculate the similarity scores.
bert embeddings:
here i had bert embeddings of each sentence using mean pooling and then calculated similarity scores.
bert+lstm:
here i had taken the output of bert and given it to a bidirectional LSTM.

Then for deploying , I had Created and activated the virtual environment:
conda create --name pdfqa_env python=3.8
conda activate pdfqa_env
Install the necessary packages:
pip install streamlit transformers pytesseract fitz
Open a terminal or command prompt in the environment, and run:
 streamlit run app.py --server.fileWatcherType none

This will open the website.Make sure that model saved by running main.ipyb must be in the same directory as app.py



