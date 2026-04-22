import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import numpy as np

# -------------------------------
# STEP 1: Sample dataset
# -------------------------------
data = {
    'email': [
        'Win money now click here http://fake.com',
        'Your bank account is blocked login now http://secure-bank.com',
        'Meeting scheduled tomorrow',
        'Project report attached',
        'Congratulations you won lottery click http://lottery-win.com',
        'Your Amazon order has been shipped https://amazon.in',
        'Update your password immediately http://phishingsite.xyz',
        'Lunch at 2pm?',
        'Verify your account now http://verify-login.com',
        'Happy birthday have a great day'
    ],
    'label': [
        'phishing','phishing','safe','safe','phishing',
        'safe','phishing','safe','phishing','safe'
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# STEP 2: URL Feature Extraction
# -------------------------------
def extract_url_features(text):
    urls = re.findall(r'https?://\S+', text)

    if len(urls) == 0:
        return [0, 0, 0, 0]

    url = urls[0]  # take first URL

    parsed = urlparse(url)

    return [
        len(url),                       # URL length
        int('https' in url),            # HTTPS or not
        int('@' in url),                # @ symbol
        parsed.netloc.count('.')        # number of dots
    ]

url_features = np.array(df['email'].apply(extract_url_features).tolist())

# -------------------------------
# STEP 3: Split
# -------------------------------
X_train_text, X_test_text, y_train, y_test, X_train_url, X_test_url = train_test_split(
    df['email'], df['label'], url_features, test_size=0.3, random_state=42
)

# -------------------------------
# STEP 4: Text Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# -------------------------------
# STEP 5: Combine Text + URL Features
# -------------------------------
X_train_final = hstack([X_train_vec, X_train_url])
X_test_final = hstack([X_test_vec, X_test_url])

# -------------------------------
# STEP 6: Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train_final, y_train)

# -------------------------------
# STEP 7: Evaluate
# -------------------------------
y_pred = model.predict(X_test_final)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# -------------------------------
# STEP 8: Custom Input (MULTI-LINE EMAIL)
# -------------------------------
while True:
    print("\nPaste full email (press ENTER twice to finish, or type 'exit'):")

    lines = []
    while True:
        line = input()
        
        if line.lower() == 'exit':
            exit()
        
        if line == "":
            break  # stop when empty line
        
        lines.append(line)

    # combine full email
    full_email = " ".join(lines)

    # text features
    user_vec = vectorizer.transform([full_email])

    # URL features
    user_url_feat = np.array(extract_url_features(full_email)).reshape(1, -1)

    # combine
    user_final = hstack([user_vec, user_url_feat])

    prediction = model.predict(user_final)

    print("\nFINAL RESULT:", prediction[0])