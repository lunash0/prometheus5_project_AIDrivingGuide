import streamlit as st
from settings import FEEDBACK_FILE_PATH
import json 

def load_feedback():
    if FEEDBACK_FILE_PATH.exists() and FEEDBACK_FILE_PATH.stat().st_size > 0:
        with open(FEEDBACK_FILE_PATH, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def save_feedback(feedback):
    with open(FEEDBACK_FILE_PATH, "w") as file:
        json.dump(feedback, file, indent=4)

def page_feedback():
    st.title("📢 Feedback")
    st.write("---")
    st.subheader("Share Your Thoughts 🤗")
    feedback = st.text_area("Your feedback", "")
    name = st.text_input("Your Name", "")
    rating = st.feedback("stars")
    sentiment_mapping = [1, 2, 3, 4, 5]

    if st.button("Submit"):
        if feedback and name:
            new_feedback = {"name": name, "text": feedback, "rating": rating}
            feedback_list = load_feedback()
            feedback_list.append(new_feedback)
            save_feedback(feedback_list)

            st.success("Thank you for your feedback! 🙌")
            st.rerun()  

    # Load and show previous feedbacks
    st.subheader("User Feedback 💬")
    feedback_list = load_feedback()
    if feedback_list:
        for feedback in feedback_list:
            if isinstance(feedback, dict):
                name = feedback.get('name', 'Anonymous')
                n_star = sentiment_mapping[feedback.get('rating', 0)]
                star = ""
                for _ in range(n_star):
                    star += '⭐️'
                st.markdown(f"- {feedback.get('text', 'No text')} - {name} {star} ({n_star}/5)")
    else:
        st.info("No feedback available yet. Be the first to leave your thoughts!")