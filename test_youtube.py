import pickle

# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)


# Test comments
comments = [
    "Congratulations! You have won a free iPhone. Click this link now!",
    "This is an amazing video, I really enjoyed it.",
    "Subscribe to my channel and win free money!",
    "The music in this video is really good.",
    "Click here to get free followers and subscribers."
]


# Predict each comment
for comment in comments:

    prediction = model.predict([comment])[0]

    print("\nComment:")
    print(comment)

    print("Prediction:")
    print(prediction)