import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from google import genai


# Load environment variables
load_dotenv()


# Theme
theme = st.radio(
    "Theme",
    ["☀️ Light Mode", "🌙 Dark Mode"],
    horizontal=True
)


# Connect to Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Page styling
if theme == "🌙 Dark Mode":
    background = "#0E1117"
    text = "#FFFFFF"
    card = "#161B22"
    border = "#30363D"
else:
    background = "#FFFFFF"
    text = "#111111"
    card = "#F8F9FA"
    border = "#DDDDDD"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {background};
        color: {text};
    }}

    .main-title {{
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 8px;
        color: {text};
    }}

    .subtitle {{
        text-align: center;
        font-size: 18px;
        margin-bottom: 35px;
        color: {text};
    }}

    .study-card {{
        padding: 28px;
        border-radius: 20px;
        border: 1px solid {border};
        background-color: {card};
        margin-top: 15px;
        margin-bottom: 25px;
    }}

    .section-title {{
        font-size: 25px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        color: {text};
    }}

    /* Buttons */
    div.stButton > button {{
        width: 100%;
        border-radius: 12px;
        padding: 12px;
        font-weight: 600;
        color: {text} !important;
        background-color: {card};
        border: 1px solid {border};
    }}

    /* PDF upload box */
    [data-testid="stFileUploader"] {{
        color: {text} !important;
    }}

    [data-testid="stFileUploader"] label {{
        color: {text} !important;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {card};
        border: 1px solid {border};
    }}

    [data-testid="stFileUploader"] small {{
        color: {text} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    '<div class="main-title">📚 Students Notes AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your personal AI study companion — upload, ask, and learn.</div>',
    unsafe_allow_html=True
)


# Upload section
st.markdown(
    """
    <div class="study-card">
        <h2>📄 Upload Your Study Notes</h2>
        <p>
            Upload a PDF containing your class notes, textbooks, or study material.
            Students Notes AI will read them and help you learn.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


pdf = st.file_uploader(
    "Choose your PDF",
    type="pdf"
)


# Process PDF
if pdf:

    reader = PdfReader(pdf)

    chunks = []
    chunk_pages = []

    chunk_size = 500


    # Extract text and create chunks
    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            for i in range(
                0,
                len(page_text),
                chunk_size
            ):

                chunk = page_text[
                    i:i + chunk_size
                ]

                chunks.append(chunk)

                chunk_pages.append(
                    page_number
                )


    # Check PDF
    if not chunks:

        st.error(
            "❌ No readable text found in this PDF."
        )

        st.stop()


    # Create embeddings
    with st.spinner(
        "🔄 Processing your notes..."
    ):

        embeddings = model.encode(
            chunks
        )


    st.success(
        "✅ Your notes are ready! Ask me a question."
    )


    # Study actions
    st.subheader(
        "🤔 What would you like to do?"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        summarize_clicked = st.button(
            "📝 Summarize"
        )


    with col2:

        explain_clicked = st.button(
            "💡 Explain"
        )


    with col3:

        quiz_clicked = st.button(
            "❓ Quiz Me"
        )


    # Summarize
    if summarize_clicked:

        try:

            with st.spinner(
                "📝 Creating your summary..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        "Summarize the following study notes "
                        "clearly and briefly:\n\n"
                        + " ".join(chunks)
                    )
                )


            st.subheader(
                "📝 Summary"
            )

            st.write(
                response.text
            )


        except Exception as e:

            st.error(
                "Gemini is temporarily unavailable. "
                "Please try again later."
            )


    # Explain
    if explain_clicked:

        try:

            with st.spinner(
                "💡 Creating an easy explanation..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        "Explain the following study notes "
                        "in very simple language for a first-year "
                        "college student. "
                        "Use simple examples where helpful:\n\n"
                        + " ".join(chunks)
                    )
                )


            st.subheader(
                "💡 Explanation"
            )

            st.write(
                response.text
            )


        except Exception as e:

            st.error(
                f"Gemini error: {e}"
            )


    # Quiz
    if quiz_clicked:

        try:

            with st.spinner(
                "❓ Creating your quiz..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        "Create exactly 5 multiple-choice questions "
                        "from the following study notes.\n\n"

                        "Return each question in this exact format:\n"
                        "QUESTION: question text\n"
                        "A: option A\n"
                        "B: option B\n"
                        "C: option C\n"
                        "D: option D\n"
                        "ANSWER: A\n\n"

                        "The ANSWER must contain only A, B, C, or D.\n"
                        "Use ONLY information from the study notes.\n\n"

                        "Study notes:\n"
                        + " ".join(chunks)
                    )
                )


            st.session_state[
                "quiz_text"
            ] = response.text

            st.session_state[
                "quiz_submitted"
            ] = False


        except Exception as e:

            st.error(
                f"Gemini error: {e}"
            )


    # Display quiz
    if "quiz_text" in st.session_state:

        st.subheader(
            "❓ Quiz Time!"
        )


        quiz_blocks = st.session_state[
            "quiz_text"
        ].split("QUESTION:")


        quiz_questions = []


        for block in quiz_blocks:

            if not block.strip():

                continue


            lines = block.strip().splitlines()

            question = lines[0].strip()

            options = {}

            correct_answer = ""


            for line in lines[1:]:

                line = line.strip()


                if line.startswith("A:"):

                    options["A"] = (
                        line[2:].strip()
                    )


                elif line.startswith("B:"):

                    options["B"] = (
                        line[2:].strip()
                    )


                elif line.startswith("C:"):

                    options["C"] = (
                        line[2:].strip()
                    )


                elif line.startswith("D:"):

                    options["D"] = (
                        line[2:].strip()
                    )


                elif line.startswith("ANSWER:"):

                    correct_answer = (
                        line.replace(
                            "ANSWER:",
                            ""
                        ).strip()
                    )


            if (
                len(options) == 4
                and correct_answer in options
            ):

                quiz_questions.append(
                    {
                        "question": question,
                        "options": options,
                        "answer": correct_answer
                    }
                )


        # Display questions
        for i, quiz in enumerate(
            quiz_questions
        ):

            st.markdown(
                f"### Question {i + 1}"
            )


            st.write(
                quiz["question"]
            )


            choices = [
                "Choose an answer",
                f"A: {quiz['options']['A']}",
                f"B: {quiz['options']['B']}",
                f"C: {quiz['options']['C']}",
                f"D: {quiz['options']['D']}"
            ]


            st.selectbox(
                "Select your answer:",
                choices,
                key=f"quiz_answer_{i}"
            )


        st.divider()


        # Submit quiz
        if st.button(
            "🚀 Submit Quiz",
            key="submit_quiz"
        ):

            score = 0

            unanswered = 0


            for i, quiz in enumerate(
                quiz_questions
            ):

                selected = st.session_state.get(
                    f"quiz_answer_{i}",
                    "Choose an answer"
                )


                if selected == "Choose an answer":

                    unanswered += 1

                    continue


                selected_letter = selected[0]


                if selected_letter == quiz["answer"]:

                    score += 1


            if unanswered > 0:

                st.warning(
                    f"⚠️ Please answer all questions "
                    f"before submitting. "
                    f"{unanswered} question(s) are unanswered."
                )


            else:

                st.session_state[
                    "quiz_submitted"
                ] = True

                st.session_state[
                    "quiz_score"
                ] = score


                st.success(
                    f"🎯 Your Score: "
                    f"{score} / {len(quiz_questions)}"
                )


        # Show results
        if st.session_state.get(
            "quiz_submitted",
            False
        ):

            st.markdown(
                "### 📊 Results"
            )


            for i, quiz in enumerate(
                quiz_questions
            ):

                selected = st.session_state.get(
                    f"quiz_answer_{i}",
                    "Choose an answer"
                )


                selected_letter = selected[0]


                if selected_letter == quiz["answer"]:

                    st.success(
                        f"Question {i + 1}: "
                        f"✅ Correct!"
                    )


                else:

                    correct = quiz["answer"]


                    st.error(
                        f"Question {i + 1}: "
                        f"❌ Incorrect. "
                        f"Correct answer: "
                        f"{correct}: "
                        f"{quiz['options'][correct]}"
                    )


    # Ask Your Notes
    st.markdown(
        '<div class="section-title">🤔 Ask Your Notes</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Ask anything about your uploaded notes. "
        "I'll find the relevant information and explain it for you."
    )


    question = st.text_input(
        "💬 Ask a question about your notes",
        placeholder="Example: What is a stack in data structures?"
    )


    # RAG question answering
    if question:

        question_embedding = model.encode(
            question
        )


        with st.spinner(
            "🔍 Finding the answer..."
        ):

            similarities = model.similarity(
                question_embedding,
                embeddings
            )


        best_index = similarities.argmax()

        best_chunk = chunks[best_index]

        best_page = chunk_pages[best_index]


        st.divider()


        try:

            with st.spinner(
                "🤖 Generating your answer..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        "Answer the student's question using ONLY "
                        "the information provided in the notes below. "
                        "If the answer is not found in the notes, "
                        "say that it is not available in the "
                        "uploaded notes.\n\n"

                        "Student's question:\n"
                        + question

                        + "\n\nRelevant notes:\n"
                        + best_chunk
                    )
                )


            st.markdown(
                "### 🤖 Answer from your notes"
            )

            st.write(
                response.text
            )

            st.info(
                "📖 Source: Page "
                + str(best_page)
            )


        except Exception as e:

            st.error(
                f"Gemini error: {e}"
            )