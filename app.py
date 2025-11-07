# app.py
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(layout="wide", page_title="Market Pulse — Sentiment Analyzer")

st.title("Market Pulse — E-commerce Sentiment Analyzer")
st.write("Upload a CSV with columns: product_name, review_text (optional: rating).")

# File uploader
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
demo_data = st.checkbox("Load demo sample data")

# If demo requested, create demo dataframe
if demo_data and uploaded is None:
    data = pd.DataFrame({
        "product_name": ["Phone A", "Phone B", "Phone A", "Phone C", "Phone B", "Headphones X"],
        "review_text": [
            "Battery lasts long and camera is good",
            "Terrible software, crashes often",
            "Okay for the price",
            "Amazing display and fast",
            "Not comfortable and cheap build",
            "Sound quality excellent but a bit pricey"
        ],
        "rating": [5,1,3,5,2,4]
    })
elif uploaded:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV or check 'Load demo sample data' to try the app.")
    st.stop()

# Validate columns
if 'review_text' not in data.columns or 'product_name' not in data.columns:
    st.error("CSV must contain at least 'product_name' and 'review_text' columns.")
    st.stop()

# Preprocess
data = data.dropna(subset=['review_text', 'product_name']).reset_index(drop=True)

# Sentiment analyzer
sid = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    scores = sid.polarity_scores(str(text))
    c = scores['compound']
    if c >= 0.05:
        label = "Positive"
    elif c <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return pd.Series([label, c])

st.info("Running sentiment analysis...")
data[['sentiment', 'compound_score']] = data['review_text'].apply(classify_sentiment)

# Sidebar filters
st.sidebar.header("Filters")
products = ["All"] + sorted(data['product_name'].unique().tolist())
sel_product = st.sidebar.selectbox("Product", products)
min_score = st.sidebar.slider("Compound score >= ", -1.0, 1.0, -1.0, 0.01)

filtered = data.copy()
if sel_product != "All":
    filtered = filtered[filtered['product_name'] == sel_product]
filtered = filtered[filtered['compound_score'] >= min_score]

# Top metrics
col1, col2, col3, col4 = st.columns(4)
total_reviews = len(filtered)
pos_pct = (filtered['sentiment'] == 'Positive').mean() * 100 if total_reviews else 0
neg_pct = (filtered['sentiment'] == 'Negative').mean() * 100 if total_reviews else 0
neu_pct = (filtered['sentiment'] == 'Neutral').mean() * 100 if total_reviews else 0

col1.metric("Reviews (filtered)", total_reviews)
col2.metric("Positive %", f"{pos_pct:.1f}%")
col3.metric("Negative %", f"{neg_pct:.1f}%")
col4.metric("Neutral %", f"{neu_pct:.1f}%")

# Sentiment distribution plot (overall or per product)
st.subheader("Sentiment Distribution")
dist = filtered['sentiment'].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
fig = px.pie(values=dist.values, names=dist.index, title="Sentiment Breakdown", hole=0.4)
st.plotly_chart(fig, use_container_width=True)

# Sentiment by product (stacked bar)
st.subheader("Sentiment by Product")
grouped = data.groupby(['product_name','sentiment']).size().reset_index(name='count')
pivot = grouped.pivot(index='product_name', columns='sentiment', values='count').fillna(0)
pivot = pivot.reindex(columns=['Positive','Neutral','Negative']).fillna(0)
pivot = pivot.reset_index()
fig2 = px.bar(pivot, x='product_name', y=['Positive','Neutral','Negative'], title="Sentiment by product", barmode='stack')
st.plotly_chart(fig2, use_container_width=True)

# Show top/bottom products by positive %
st.subheader("Top/Bottom products (by positive %)")
summary = data.groupby('product_name').agg(
    total_reviews=('review_text','count'),
    positive=('sentiment', lambda s: (s=='Positive').sum())
).reset_index()
summary['positive_pct'] = summary['positive'] / summary['total_reviews'] * 100
top = summary.sort_values('positive_pct', ascending=False).head(5)
bottom = summary.sort_values('positive_pct', ascending=True).head(5)

colA, colB = st.columns(2)
with colA:
    st.markdown("**Top 5 products**")
    st.table(top[['product_name','positive_pct','total_reviews']].rename(columns={'product_name':'Product','positive_pct':'Positive %','total_reviews':'Reviews'}).style.format({'Positive %':'{:.1f}'}))
with colB:
    st.markdown("**Bottom 5 products**")
    st.table(bottom[['product_name','positive_pct','total_reviews']].rename(columns={'product_name':'Product','positive_pct':'Positive %','total_reviews':'Reviews'}).style.format({'Positive %':'{:.1f}'}))

# Show sample reviews
st.subheader("Sample Reviews (filtered)")
display_cols = ['product_name','review_text','sentiment','compound_score']
if 'rating' in filtered.columns:
    display_cols.insert(2, 'rating')
st.dataframe(filtered[display_cols].sort_values('compound_score', ascending=False).reset_index(drop=True))

# Wordcloud for selected sentiment
st.subheader("Wordcloud")
wc_sent = st.selectbox("Wordcloud for sentiment", ['All','Positive','Negative','Neutral'])
if wc_sent == 'All':
    texts = " ".join(data['review_text'].astype(str).tolist())
else:
    texts = " ".join(data.loc[data['sentiment']==wc_sent,'review_text'].astype(str).tolist())

if texts.strip()=="":
    st.info("No text for wordcloud.")
else:
    wc = WordCloud(width=800, height=400, collocations=False).generate(texts)
    fig_wc = plt.figure(figsize=(12,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)

# Allow download of results
out = io.StringIO()
data.to_csv(out, index=False)
st.download_button("Download analyzed results as CSV", data=out.getvalue(), file_name="analyzed_reviews.csv", mime="text/csv")
