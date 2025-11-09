import streamlit as st
import pandas as pd
from datetime import datetime
from TextAsData import predict_sentiment
from feedback_stream import stream_feedback
import plotly.graph_objects as go
import string

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

st.set_page_config(page_title="T-Mobile Customer Happiness Index", layout="wide")

# Inject CSS to enforce dark theme + neon colors more reliably
st.markdown(
    """
    <style>
    /* Overall background and text */
    body, .css-k1vhr4 e1fqkh3o3, .block-container {
        background-color: #000000 !important;
        color: #00FFFF !important;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #00FFFF !important;
    }
    /* Headings and main text */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #00FFFF !important;
    }
    /* Links */
    a {
        color: #00FFFF !important;
    }
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    ::-webkit-scrollbar-thumb {
        background: #00FFFF;
        border-radius: 4px;
    }
    /* Progress bar neon gradient */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00FFEF, #00FFE7) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 T-Mobile Customer Happiness Index")

data = {
    "timestamp": [],
    "feedback": [],
    "sentiment_score": [],
    "sentiment_label": []
}

def sentiment_label(score, threshold=0.5):
    return "Positive" if score >= threshold else "Negative"

feedback_stream = stream_feedback("feedback_labeled.csv")

placeholder = st.empty()

for feedback in feedback_stream:
    cleaned_feedback = clean_text(feedback)
    score = predict_sentiment(cleaned_feedback)
    label = sentiment_label(score)

    data["timestamp"].append(datetime.now())
    data["feedback"].append(feedback)
    data["sentiment_score"].append(score)
    data["sentiment_label"].append(label)

    df = pd.DataFrame(data)
    total = len(df)
    positive = (df["sentiment_label"] == "Positive").sum()
    negative = total - positive
    avg_sentiment = df["sentiment_score"].mean()

    with placeholder.container():
        st.markdown(f"<h2 style='color:#00FFFF;'>💜 Customer Happiness Index: {avg_sentiment:.2f} / 1.0</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"<h4 style='color:#FF00FF;'>Summary</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>Total Feedback Processed: <b>{total}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#00FF00;'>Positive Feedback: <b>{positive} ({positive/total:.0%})</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#FF4444;'>Negative Feedback: <b>{negative} ({negative/total:.0%})</b></p>", unsafe_allow_html=True)

            # Progress bar styled neon blue
            st.progress(avg_sentiment)

            # Recent feedback with colors
            st.markdown("<h4 style='color:#00FFFF;'>🗣️ Recent Feedback</h4>", unsafe_allow_html=True)
            for idx, row in df.tail(5).iterrows():
                color = "#00FF00" if row['sentiment_label'] == "Positive" else "#FF4444"
                st.markdown(f"<p style='color:{color};font-weight:bold;'>{row['sentiment_label']} ({row['sentiment_score']:.2f}): {row['feedback']}</p>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<h4 style='color:#FF00FF;'>Sentiment Distribution</h4>", unsafe_allow_html=True)
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative'],
                values=[positive, negative],
                hole=0.4,
                marker=dict(colors=['#00FF00', '#FF4444']),
                hoverinfo="label+percent+value"
            )])
            fig_pie.update_layout(
                paper_bgcolor='black',
                font=dict(color='white'),
                legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(f"<h4 style='color:#00FFFF;'>Sentiment Score Over Time</h4>", unsafe_allow_html=True)
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['sentiment_score'],
                mode='lines+markers',
                line=dict(color='#00FFFF', width=3),
                marker=dict(size=7)
            ))
            fig_line.update_layout(
                xaxis_title='Time',
                yaxis_title='Sentiment Score',
                yaxis_range=[0, 1],
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_line, use_container_width=True)
