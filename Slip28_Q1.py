from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(newsgroups_train.data, newsgroups_train.target)

def predict_news_category(text):
    predicted_index = model.predict([text])[0]
    return newsgroups_train.target_names[predicted_index]

news_text = """
NASA announced a new mission to explore the outer planets.
The mission will use advanced robotics and AI to study Jupiter and Saturn.
"""

predicted_category = predict_news_category(news_text)
print("Predicted Category:", predicted_category)
test_accuracy = model.score(newsgroups_test.data, newsgroups_test.target)
print(f"Model accuracy on test set: {test_accuracy:.2f}")
