import streamlit as st
import pickle
import pandas as pd

from textblob import TextBlob
from youtube_comments import get_comments


# -----------------------------
# Load Spam Detection Model
# -----------------------------

with open("spam_model.pkl", "rb") as file:
    spam_model = pickle.load(file)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Social Media Comment Analyzer",
    page_icon="💬",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("💬 Social Media Comment Analyzer")

st.write(
    "Fetch YouTube comments and detect spam "
    "and sentiment using Machine Learning."
)


# -----------------------------
# YouTube Section
# -----------------------------

st.header("📺 Analyze YouTube Comments")

video_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

number_of_comments = st.slider(
    "Number of comments",
    min_value=10,
    max_value=500,
    value=20
)


# -----------------------------
# Sentiment Function
# -----------------------------

def get_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"


# -----------------------------
# Analyze Button
# -----------------------------

if st.button("Fetch & Analyze Comments"):

    if video_url.strip() == "":

        st.warning("Please enter a YouTube video URL.")

    else:

        try:

            with st.spinner("Fetching YouTube comments..."):

                comments = get_comments(
                    video_url,
                    number_of_comments
                )


            if len(comments) == 0:

                st.warning(
                    "No comments found for this video."
                )

            else:

                results = []


                # -----------------------------
                # Analyze each comment
                # -----------------------------

                for item in comments:

                    comment_text = item["comment"]

                    prediction = spam_model.predict(
                        [comment_text]
                    )[0]


                    if prediction == "spam":

                        sentiment = "Not Applicable"

                    else:

                        sentiment = get_sentiment(
                            comment_text
                        )


                    results.append({
                        "Author": item["author"],
                        "Comment": comment_text,
                        "Spam": (
                            "Spam"
                            if prediction == "spam"
                            else "Not Spam"
                        ),
                        "Sentiment": sentiment,
                        "Likes": item["likes"]
                    })


                # -----------------------------
                # Create DataFrame
                # -----------------------------

                df = pd.DataFrame(results)


                # -----------------------------
                # Statistics
                # -----------------------------

                total = len(df)

                spam_count = (
                    df["Spam"] == "Spam"
                ).sum()

                not_spam_count = (
                    df["Spam"] == "Not Spam"
                ).sum()


                positive_count = (
                    df["Sentiment"] == "Positive"
                ).sum()

                negative_count = (
                    df["Sentiment"] == "Negative"
                ).sum()

                neutral_count = (
                    df["Sentiment"] == "Neutral"
                ).sum()


                # -----------------------------
                # Display Metrics
                # -----------------------------

                st.subheader("📊 Analysis Summary")


                col1, col2, col3, col4 = st.columns(4)


                col1.metric(
                    "Total Comments",
                    total
                )

                col2.metric(
                    "Spam",
                    spam_count
                )

                col3.metric(
                    "Not Spam",
                    not_spam_count
                )

                col4.metric(
                    "Positive",
                    positive_count
                )


                col5, col6 = st.columns(2)


                col5.metric(
                    "Negative",
                    negative_count
                )

                col6.metric(
                    "Neutral",
                    neutral_count
                )


                # -----------------------------
                # Display Results
                # -----------------------------

                st.subheader("💬 Comment Analysis")

                st.dataframe(
                    df,
                    use_container_width=True
                )


                # -----------------------------
                # Charts
                # -----------------------------

                st.subheader("📈 Spam Distribution")

                spam_chart = df["Spam"].value_counts()

                st.bar_chart(spam_chart)


                st.subheader("😊 Sentiment Distribution")

                sentiment_chart = (
                    df["Sentiment"]
                    .value_counts()
                )

                st.bar_chart(sentiment_chart)


                # -----------------------------
                # Download Results
                # -----------------------------

                csv = df.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv,
                    file_name="youtube_comment_analysis.csv",
                    mime="text/csv"
                )


        except Exception as e:

            st.error(
                f"Error: {e}"
            )