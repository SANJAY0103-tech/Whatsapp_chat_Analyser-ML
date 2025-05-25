import streamlit as st
import preprocessor
import helper_updated as helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')

st.set_page_config(layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file (.txt)")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green', marker='o')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black', marker='.')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)

        # Heatmap
        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

        # Most Busy Users
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)
        st.title("Most Common Words")
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='skyblue')
        ax.invert_yaxis()  # So the most common is at the top
        plt.tight_layout()
        st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f%%")
            st.pyplot(fig)

        # Sentiment Analysis
        st.title("Sentiment Analysis")
        sentiment_counts, sentiment_df = helper.sentiment_analysis(selected_user, df)
        fig, ax = plt.subplots()
        colors = ['green', 'red', 'gray']
        # Match color length to sentiment categories dynamically
        colors = colors[:len(sentiment_counts)]
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(sentiment_df[['user', 'message', 'sentiment']].tail(10))

        # Message Classification
        st.title("Message Classification")
        category_counts, classified_df = helper.message_classification(selected_user, df)
        fig, ax = plt.subplots()
        ax.pie(category_counts.values, labels=category_counts.index, autopct="%0.1f%%")
        st.pyplot(fig)
        st.dataframe(classified_df[['user', 'message', 'category']].tail(10))

        # # Chat Summary
        # st.title("Chat Summary")
        # summary_sentences = helper.summarize_chats(selected_user, df)
        # for i, sentence in enumerate(summary_sentences):
        #     st.write(f"{i+1}. {sentence}")

        # User Personality Insights
        if selected_user == 'Overall':
            st.title("User Personality Insights")
            personality_data = helper.user_personality_insights(df)
            st.dataframe(personality_data)

else:
    st.warning("📂 Please upload a WhatsApp chat text file to begin analysis.")
