import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional ML imports (loaded only when needed)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Instagram Usage Analytics", layout="wide")

st.title("📊 Instagram Usage & Lifestyle Analytics Dashboard")
st.markdown("Interactive UI built from the original analysis file")

# -------------------- LOAD DATA (NO CACHE - SAFE MODE) --------------------

#1111 df = pd.read_csv("instagram_usage_lifestyle.csv")

#2222 import gdown
# url = "https://drive.google.com/uc?id=1mU7OYUaC2Dl8pOWUnNoeFvx5uBQ9sJC9"
# output = "instagram_usage_lifestyle.csv"
# gdown.download(url, output, quiet=False)
# df = pd.read_csv(output)
import kagglehub
import os

#@st.cache_data(show_spinner=True)
def load_data():
    # Download dataset from Kaggle (runs only first time)
    path = kagglehub.dataset_download("rockyt07/social-media-user-analysis")

    csv_file = os.path.join(path, "instagram_usage_lifestyle.csv")

    # Load limited rows to prevent memory crash
    df = pd.read_csv(csv_file, nrows=100000)

    return df

df = load_data()


# -------------------- DROP COLUMNS (FROM ORIGINAL FILE) --------------------
drop_cols = [
    "diet_quality","alcohol_frequency","body_mass_index","blood_pressure_systolic",
    "blood_pressure_diastolic","daily_steps_count","volunteer_hours_per_month",
    "income_level","relationship_status","has_children","exercise_hours_per_week",
    "self_reported_happiness","smoking","user_engagement_score","subscription_status",
    "biometric_login_used","two_factor_auth_enabled","app_name","user_id"
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

# -------------------- FEATURE ENGINEERING --------------------
df['hours'] = df['daily_active_minutes_instagram'] / 60

df['total_time_per_day_mins'] = (
    df['time_on_feed_per_day'] +
    df['time_on_explore_per_day'] +
    df['time_on_messages_per_day'] +
    df['time_on_reels_per_day']
)

df['usage_level'] = pd.cut(df['total_time_per_day_mins'],
                           bins=[0,1,3,6,24],
                           labels=['Low','Medium','High','Extreme'])

df['engagement_score'] = (
    df['likes_given_per_day'] + df['comments_written_per_day'] +
    df['dms_sent_per_week']/7 + df['dms_received_per_week']/7 +
    df['posts_created_per_week']/7 + df['reels_watched_per_day']
)

df['lifestyle_score'] = (
    df['sleep_hours_per_night'] + df['hobbies_count'] + df['social_events_per_month']
) - df['total_time_per_day_mins']

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ User Controls")

selected_cols = st.sidebar.multiselect(
    "Select Columns to Display",
    df.columns.tolist(),
    default=['age','gender','country','hours','total_time_per_day_mins']
)

row_limit = st.sidebar.slider("Rows to Display", 10, 500, 50)

run_clustering = st.sidebar.checkbox("Enable User Persona Clustering (ML)", value=False)

# -------------------- FILTERED DATA VIEW --------------------
st.subheader("📄 Filtered Dataset Preview")
st.dataframe(df[selected_cols].head(row_limit))

# -------------------- BASIC STATS --------------------
st.subheader("📈 Summary Statistics")
st.write(df[selected_cols].describe())

# -------------------- AVG AGE BY GENDER --------------------
st.subheader("👥 Average Age by Gender")
avg_age = df.groupby('gender')['age'].mean()

fig1, ax1 = plt.subplots()
ax1.scatter(avg_age.index, avg_age.values)
ax1.set_xlabel("Gender")
ax1.set_ylabel("Average Age")
st.pyplot(fig1)

# -------------------- AVG TIME BY AGE --------------------
st.subheader("⏳ Average Instagram Time by Age")
avg_time = df.groupby("age")['hours'].mean()

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(avg_time.index, avg_time.values)
ax2.set_xlabel("Age")
ax2.set_ylabel("Hours")
st.pyplot(fig2)

# -------------------- CORRELATION HEATMAP --------------------
st.subheader("🔥 Lifestyle Impact Correlation Heatmap")
impact_cols = ['hours','sleep_hours_per_night','weekly_work_hours','perceived_stress_score']

fig3, ax3 = plt.subplots(figsize=(7,5))
sns.heatmap(df[impact_cols].corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# -------------------- USAGE DISTRIBUTION --------------------
st.subheader("📊 Usage Distribution")

col1, col2 = st.columns(2)

with col1:
    fig4, ax4 = plt.subplots()
    sns.histplot(df['total_time_per_day_mins'], bins=30, ax=ax4)
    st.pyplot(fig4)

with col2:
    fig5, ax5 = plt.subplots()
    sns.boxplot(x='gender', y='total_time_per_day_mins', data=df, ax=ax5)
    st.pyplot(fig5)

# -------------------- ENGAGEMENT SCORE --------------------
st.subheader("❤️ Engagement Score Distribution")
fig6, ax6 = plt.subplots()
sns.histplot(x='engagement_score', data=df, ax=ax6)
st.pyplot(fig6)

# -------------------- LIFESTYLE SCORE --------------------
st.subheader("🌱 Lifestyle Score Distribution")
fig7, ax7 = plt.subplots()
sns.histplot(x='lifestyle_score', data=df, ax=ax7)
st.pyplot(fig7)

# -------------------- EXTENDED CORRELATION --------------------
st.subheader("🧠 Detailed Correlation Analysis")

corr_cols = ['hours','sleep_hours_per_night','perceived_stress_score','weekly_work_hours',
             'hobbies_count','social_events_per_month','books_read_per_year',
             'reels_watched_per_day','stories_viewed_per_day','likes_given_per_day','comments_written_per_day']

fig8, ax8 = plt.subplots(figsize=(11,9))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax8)
st.pyplot(fig8)

# -------------------- USER PERSONA CLUSTERING (OPTIONAL) --------------------
if run_clustering:
    st.subheader("👤 User Persona Clustering")

    features = df[['hours','engagement_score','lifestyle_score']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['user_persona'] = kmeans.fit_predict(scaled_features)

    sample_df = df.sample(10000)

    fig9, ax9 = plt.subplots()
    sns.scatterplot(x='hours', y='engagement_score', hue='user_persona', data=sample_df, ax=ax9)
    st.pyplot(fig9)

# -------------------- COUNTRY PIE CHART --------------------
st.subheader("🌍 Country Distribution")

country_counts = df['country'].value_counts()

fig10, ax10 = plt.subplots(figsize=(3,3))
ax10.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%',textprops={'fontsize':5})
ax10.set_title("Country Distribution")
st.pyplot(fig10)

# -------------------- KDE PLOTS --------------------
st.subheader("📉 Density Plots")

col3, col4 = st.columns(2)

with col3:
    fig11, ax11 = plt.subplots()
    sns.kdeplot(df['weekly_work_hours'], fill=True, ax=ax11)
    st.pyplot(fig11)

with col4:
    fig12, ax12 = plt.subplots()
    sns.kdeplot(df['sleep_hours_per_night'], fill=True, ax=ax12)
    st.pyplot(fig12)

st.success("✅ Dashboard Loaded Successfully (Safe Mode)")

