# chatbot_crushers
Project Structure
1. Data Preparation
Data Source:

A text file named SamsungDialog.txt containing the conversation between a customer and a sales agent.
Data Format:

The conversation is divided into two columns: Customer Questions and Sales Agent Answers.
2. Libraries Used
nltk: For natural language processing, particularly for text preprocessing tasks such as tokenization and lemmatization.
TfidfVectorizer: To convert the text data into numerical vectors based on the TF-IDF scheme.
cosine similarity: To compute similarity scores between user queries and existing responses.
pandas: For data manipulation and organization.

3. Text Preprocessing
A function was created to preprocess the text, including the following steps:

Tokenization: Splitting the text into individual words.
Lemmatization: Reducing words to their base or root form.

4. Vectorization
Corpus Vectorization: The entire corpus (set of documents) is vectorized using TfidfVectorizer.
Input Vectorization: User input is preprocessed and vectorized similarly to the corpus.

6. Similarity Scoring
Cosine Similarity: Used to calculate the similarity scores between the user's query vector and the vectors of the responses in the corpus.

8. Bot Greetings and Farewells
Greeting Statements: Predefined responses for user greetings.
Farewell Statements: Predefined responses for user farewells.

10. Conversation History
Another file is created to store the history of the conversation for tracking and analysis.
