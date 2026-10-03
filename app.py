
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="EduPro Analytics",
    page_icon="📚",
    layout="wide"
)

st.title("📚 EduPro: Course Demand & Revenue Analytics")
st.caption(
    "Explore enrollment trends, course performance, revenue patterns, "
    "and predictive estimates."
)

@st.cache_data
def load_data():
    df = pd.read_csv("edupro_monthly_model_data.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    comparison = pd.read_csv("edupro_model_comparison.csv")
    return df, comparison

try:
    df, comparison = load_data()
except FileNotFoundError:
    st.error(
        "CSV files not found. Place both CSV files in the same folder as app.py."
    )
    st.stop()

numeric_features = [
    "PreviousMonthEnrollments",
    "TwoMonthsAgoEnrollments",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "AvgTeacherExperience",
    "AvgTeacherRating",
    "TeacherCount"
]

categorical_features = [
    "CourseCategory",
    "CourseType",
    "CourseLevel"
]

features = numeric_features + categorical_features

# Chronological split: train before October, test October onward
train = df[df["Month"] < "2025-10-01"].copy()
test = df[df["Month"] >= "2025-10-01"].copy()

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

@st.cache_resource
def train_models(train_data):
    enrollment_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    revenue_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    enrollment_model.fit(
        train_data[features], train_data["EnrollmentCount"]
    )
    revenue_model.fit(
        train_data[features], train_data["CourseRevenue"]
    )

    return enrollment_model, revenue_model

enrollment_model, revenue_model = train_models(train)

# Sidebar filters
st.sidebar.header("Dashboard Filters")

categories = sorted(df["CourseCategory"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Course category",
    categories,
    default=categories
)

filtered = df[df["CourseCategory"].isin(selected_categories)].copy()

if filtered.empty:
    st.warning("Select at least one course category.")
    st.stop()

# Summary metrics
total_enrollments = int(filtered["EnrollmentCount"].sum())
total_revenue = filtered["CourseRevenue"].sum()
course_count = filtered["CourseID"].nunique()
average_enrollments = filtered["EnrollmentCount"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Courses", f"{course_count}")
c2.metric("Recorded enrollments", f"{total_enrollments:,}")
c3.metric("Recorded revenue", f"₹{total_revenue:,.2f}")
c4.metric("Avg. monthly enrollments", f"{average_enrollments:.1f}")

overview, trends, categories_tab, models_tab, prediction = st.tabs([
    "Overview",
    "Monthly Trends",
    "Category Analysis",
    "Model Performance",
    "Predict"
])

with overview:
    st.subheader("Course Performance")

    course_summary = (
        filtered.groupby("CourseID", as_index=False)
        .agg(
            EnrollmentCount=("EnrollmentCount", "sum"),
            CourseRevenue=("CourseRevenue", "sum")
        )
        .sort_values("EnrollmentCount", ascending=False)
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Top courses by recorded enrollments**")
        st.dataframe(course_summary.head(10), use_container_width=True)

    with right:
        st.markdown("**Top courses by recorded revenue**")
        top_revenue = course_summary.sort_values(
            "CourseRevenue", ascending=False
        ).head(10)

        fig = px.bar(
            top_revenue,
            x="CourseRevenue",
            y="CourseID",
            orientation="h",
            title="Top 10 courses by revenue"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with trends:
    st.subheader("Monthly Enrollment and Revenue Trends")

    monthly = (
        filtered.groupby("Month", as_index=False)
        .agg(
            EnrollmentCount=("EnrollmentCount", "sum"),
            CourseRevenue=("CourseRevenue", "sum")
        )
        .sort_values("Month")
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly,
            x="Month",
            y="EnrollmentCount",
            markers=True,
            title="Monthly enrollments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            monthly,
            x="Month",
            y="CourseRevenue",
            markers=True,
            title="Monthly revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(monthly, use_container_width=True)

with categories_tab:
    st.subheader("Compare Course Categories")

    category_summary = (
        filtered.groupby("CourseCategory", as_index=False)
        .agg(
            EnrollmentCount=("EnrollmentCount", "sum"),
            CourseRevenue=("CourseRevenue", "sum")
        )
        .sort_values("CourseRevenue", ascending=False)
    )

    metric_choice = st.radio(
        "Compare categories by",
        ["EnrollmentCount", "CourseRevenue"],
        horizontal=True
    )

    fig = px.bar(
        category_summary,
        x="CourseCategory",
        y=metric_choice,
        title=f"Category comparison: {metric_choice}"
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(category_summary, use_container_width=True)

with models_tab:
    st.subheader("Model Performance")

    st.write(
        "MAE and RMSE measure prediction error. Lower values are better. "
        "R² measures how well a model explains variation; values closer "
        "to 1 are generally better."
    )

    st.dataframe(comparison, use_container_width=True)

    st.info(
        "Important: enrollment models had negative R² on the held-out "
        "months, so their predictive reliability is limited. Revenue "
        "performance looks stronger in this test, but it is based on "
        "only one year of data and needs more validation."
    )

with prediction:
    st.subheader("Estimate Demand and Revenue for a Course")
    st.write(
        "Enter course and instructor details to generate model estimates. "
        "These are exploratory predictions, not guaranteed outcomes."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox("Course category", categories)
            course_type = st.selectbox("Course type", sorted(df["CourseType"].unique()))
            level = st.selectbox("Course level", sorted(df["CourseLevel"].unique()))
            price = st.number_input(
                "Course price (₹)", min_value=0.0, value=299.0
            )
            duration = st.number_input(
                "Course duration", min_value=1.0, value=6.0
            )

        with col2:
            course_rating = st.slider(
                "Course rating", 1.0, 5.0, 4.0, 0.1
            )
            teacher_experience = st.number_input(
                "Instructor experience (years)",
                min_value=0.0,
                value=3.0
            )
            teacher_rating = st.slider(
                "Instructor rating", 1.0, 5.0, 4.0, 0.1
            )
            teacher_count = st.number_input(
                "Number of instructors", min_value=1, value=1
            )
            previous_enrollments = st.number_input(
                "Previous month enrollments",
                min_value=0,
                value=50
            )
            two_months_enrollments = st.number_input(
                "Enrollments two months ago",
                min_value=0,
                value=45
            )

        submitted = st.form_submit_button("Generate estimates")

    if submitted:
        input_data = pd.DataFrame([{
            "PreviousMonthEnrollments": previous_enrollments,
            "TwoMonthsAgoEnrollments": two_months_enrollments,
            "CoursePrice": price,
            "CourseDuration": duration,
            "CourseRating": course_rating,
            "AvgTeacherExperience": teacher_experience,
            "AvgTeacherRating": teacher_rating,
            "TeacherCount": teacher_count,
            "CourseCategory": category,
            "CourseType": course_type,
            "CourseLevel": level
        }])

        estimated_enrollments = max(
            0, int(round(enrollment_model.predict(input_data)[0]))
        )
        estimated_revenue = max(
            0, revenue_model.predict(input_data)[0]
        )

        p1, p2 = st.columns(2)
        p1.metric("Estimated enrollments", f"{estimated_enrollments:,}")
        p2.metric("Estimated revenue", f"₹{estimated_revenue:,.2f}")

        st.caption(
            "Predictions are based on historical 2025 data. "
            "Actual results may differ."
        )

st.divider()
st.caption("EduPro Predictive Analytics | Academic project")