"""
Frontend
"""
import os
import yaml
import streamlit as st
from .src.evaluate.evaluate import evaluate_input

config_path = '/config/params.yml'


def main_page() -> None:
    """
    project description page
    :return: None
    """
    st.markdown('# Project description')
    st.title('MLOps project:  Student Performance Prediction')
    st.write(
        """
        The university accumulates a large amount of data about the activities 
        and results of students that can be used to improve the educational process.
        This system allows you to predict academic debts of students and improve the quality of education.
        """)

    st.markdown(
        """
        Parameter description:
        1. Study_Year: Year of study
        2. Semester: Semester of study
        3. Control_type: Type of knowledge control
        4. Choice of discipline: Elective discipline
        5. Faculty: Faculty of study
        6. Gender: Students gender
        7. Citizenship: Students citizenship
        8. Type of entrance exam: Type of entrance exam
        9. Exam subject 1: subject of the first entrance examination
        10. Exam subject 2: subject of the second entrance examination
        11. Exam subject 3: subject of the third entrance examination
        12. Score on exam 1: first exam grade
        13. Score on exam 2: second exam grade
        14. Score on exam 3: third exam grade
        15. Type of olympiad: Type of olympiad
        16. Number of region: Number of region
        17. Kurs: Course of Study
        18. Your study status: Your study status
        19. Training period in days: how many days did you study
        """)


def prediction() -> None:
    """
    Getting predictions by entering data
    :return: None
    """
    st.markdown("# Prediction")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # checking for a trained model
    if os.path.exists(config["models"]["catboost"]):
        evaluate_input(data_unique_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Please train the model first")


def main() -> None:
    """
    Building a pipeline in one block
    """
    page_names_to_funcs = {
        "Project description": main_page,
        "Prediction": prediction}

    selected_page = st.sidebar.selectbox("Select an item", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
