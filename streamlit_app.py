"""
סוכן ניתוח נתונים — מועדון טניס שולחן מבואות החרמון / קורטדו
מבוסס על AI Data Analysis Agent מתוך awesome-llm-apps (Apache-2.0),
עם התאמות לעברית, תמיכה באקסל, ומעבר ל-Gemini (שכבת חינם) במקום OpenAI.
"""

import csv
import tempfile

import pandas as pd
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

st.set_page_config(page_title="סוכן ניתוח נתונים", page_icon="📊", layout="wide")

# ---------- עיצוב RTL ----------
st.markdown(
    """
    <style>
      .stApp { direction: rtl; }
      section[data-testid="stSidebar"] { direction: rtl; }
      textarea, input { direction: rtl; text-align: right; }
      .stDataFrame { direction: ltr; }
    </style>
    """,
    unsafe_allow_html=True,
)


def preprocess_and_save(file):
    """קורא את הקובץ שהועלה, מנרמל טיפוסים ושומר כ-CSV זמני עבור DuckDB."""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8-sig", na_values=["NA", "N/A", "missing", "-", ""])
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing", "-", ""])
        else:
            st.error("פורמט לא נתמך. אפשר להעלות CSV או Excel בלבד.")
            return None, None, None

        # ניקוי שמות עמודות (רווחים מיותרים שוברים שאילתות SQL)
        df.columns = [str(c).strip() for c in df.columns]

        for col in df.columns:
            low = col.lower()
            if "date" in low or "תאריך" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            elif df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")
        return None, None, None


st.title("📊 סוכן ניתוח נתונים")
st.caption("מעלים קובץ CSV או אקסל, ושואלים שאלות בעברית רגילה — בלי SQL. רץ על Gemini, בחינם.")

# ---------- מפתח API ----------
with st.sidebar:
    st.header("הגדרות")
    try:
        default_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        default_key = ""

    if default_key:
        st.session_state.google_key = default_key
        st.success("מפתח Gemini נטען מההגדרות ✅")
    else:
        google_key = st.text_input("מפתח Gemini:", type="password")
        if google_key:
            st.session_state.google_key = google_key
            st.success("המפתח נשמר לסשן הזה ✅")
        else:
            st.warning("צריך להזין מפתח Gemini כדי להתחיל.")
            st.caption("מפתח חינמי: aistudio.google.com/apikey")

    model_id = st.selectbox(
        "מודל",
        ["gemini-3.5-flash", "gemini-2.5-pro"],
        index=0,
        help="flash מהיר ובעל מכסה חינמית גבוהה. pro חזק יותר לניתוחים מורכבים, מכסה חינמית נמוכה יותר.",
    )

uploaded_file = st.file_uploader("העלאת קובץ CSV או Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None and st.session_state.get("google_key"):
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        with st.expander(f"תצוגת הנתונים ({len(df):,} שורות, {len(columns)} עמודות)", expanded=True):
            st.dataframe(df, use_container_width=True)
            st.write("עמודות:", ", ".join(columns))

        duckdb_tools = DuckDbTools()
        duckdb_tools.load_local_csv_to_table(path=temp_path, table="uploaded_data")

        data_analyst_agent = Agent(
            model=Gemini(id=model_id, api_key=st.session_state.google_key),
            tools=[duckdb_tools, PandasTools()],
            system_message=(
                "You are an expert data analyst. Use the 'uploaded_data' table to answer user queries. "
                "Generate SQL queries using DuckDB tools to solve the user's query. "
                f"The table columns are: {', '.join(columns)}. "
                "Column names may be in Hebrew — always wrap column names in double quotes in SQL. "
                "ALWAYS answer in Hebrew, clearly and concisely. "
                "Show the number you calculated, and add one short sentence of interpretation. "
                "If the question is ambiguous, state the assumption you made."
            ),
            markdown=True,
        )

        st.subheader("שאלה על הנתונים")
        user_query = st.text_area(
            "מה תרצה לדעת?",
            placeholder="לדוגמה: איזה נושא פוסט הביא את החשיפה הממוצעת הגבוהה ביותר?",
            height=90,
        )

        if st.button("שלח שאלה", type="primary"):
            if not user_query.strip():
                st.warning("צריך לכתוב שאלה.")
            else:
                try:
                    with st.spinner("מנתח..."):
                        response = data_analyst_agent.run(user_query)
                        content = getattr(response, "content", None) or str(response)
                    st.markdown(content)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "quota" in msg.lower():
                        st.error("נגמרה המכסה החינמית להיום. אפשר לנסות שוב מחר, או לעבור למודל flash.")
                    else:
                        st.error(f"שגיאה בהרצת הסוכן: {e}")
                        st.info("נסה לנסח את השאלה אחרת, או לוודא שהקובץ תקין.")

elif uploaded_file is not None:
    st.info("קודם צריך להזין מפתח Gemini בסרגל הצד.")
