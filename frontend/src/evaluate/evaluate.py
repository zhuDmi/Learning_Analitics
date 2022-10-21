"""
Drawing sliders and buttons for data entry
with further prediction based on the entered values
"""

import json
import requests
import streamlit as st


def evaluate_input(data_unique_path: str, endpoint: object) -> None:
    """
    Getting input data by typing in UI -> displaying result
    :param data_unique_path: path to unique values
    :param endpoint: endpoint
    :return: None
    """
    with open(data_unique_path) as file:
        unique_df = json.load(file)

    # data entry fields, use unique values
    st_year = st.sidebar.selectbox('Study_Year', unique_df['ST_YEAR'])
    semester = st.sidebar.slider('Semester',
                                 min_value=min(unique_df['SEMESTER']),
                                 max_value=max(unique_df['SEMESTER']),
                                 step=1)
    type_name = st.sidebar.selectbox('Control_type', unique_df['TYPE_NAME'])
    choice = st.sidebar.selectbox('Choice of discipline', unique_df['CHOICE'])
    disc_dep = st.sidebar.selectbox('Faculty', unique_df['DISC_DEP'])
    stud_gender = st.sidebar.selectbox('Gender', unique_df['STUD_GENDER'])
    citizenship = st.sidebar.selectbox('Citizenship', unique_df['CITIZENSHIP'])
    exam_type = st.sidebar.selectbox('Type of entrance exam', unique_df['EXAM_TYPE'])
    exam_sbj_1 = st.sidebar.selectbox('Exam subject 1', unique_df['EXAM_SUBJECT_1'])
    exam_sbj_2 = st.sidebar.selectbox('Exam subject 2', unique_df['EXAM_SUBJECT_2'])
    exam_sbj_3 = st.sidebar.selectbox('Exam subject 3', unique_df['EXAM_SUBJECT_3'])
    admitted_exam_1 = st.sidebar.selectbox('Score on exam 1', unique_df['ADMITTED_EXAM_1'])
    admitted_exam_2 = st.sidebar.selectbox('Score on exam 2', unique_df['ADMITTED_EXAM_2'])
    admitted_exam_3 = st.sidebar.selectbox('Score on exam 3', unique_df['ADMITTED_EXAM_3'])
    subj_prize_level = st.sidebar.selectbox('Type of olympiad', unique_df['ADMITTED_SUBJECT_PRIZE_LEVEL'])
    region = st.sidebar.selectbox('Number of region', unique_df['REGION_ID'])
    kurs = st.sidebar.slider('Kurs',
                             min_value=min(unique_df['KURS']),
                             max_value=max(unique_df['KURS']),
                             step=1)
    priznak = st.sidebar.selectbox('Your study status', unique_df['PRIZNAK'])
    training_period = st.sidebar.slider('Training period in days',
                                        min_value=min(unique_df['Training_period']),
                                        max_value=max(unique_df['Training_period']))

    dict_data = {
        'Study_Year': st_year,
        'Semester': semester,
        'Control_type': type_name,
        'Choice of discipline': choice,
        'Faculty': disc_dep,
        'Gender': stud_gender,
        'Citizenship': citizenship,
        'Type of entrance exam': exam_type,
        'Exam subject 1': exam_sbj_1,
        'Exam subject 2': exam_sbj_2,
        'Exam subject 3': exam_sbj_3,
        'Score on exam 1': admitted_exam_1,
        'Score on exam 2': admitted_exam_2,
        'Score on exam 3': admitted_exam_3,
        'Type of olympiad': subj_prize_level,
        'Number of region': region,
        'Kurs': kurs,
        'Your study status': priznak,
        'Training period in days': training_period}

    st.write(
        f"""### Student data:\n
        1. Study_Year: {dict_data['Study_Year']}
        2. Semester: {dict_data['Semester']}
        3. Control_type: {dict_data['Control_type']}
        4. Choice of discipline: {dict_data['Choice of discipline']}
        5. Faculty: {dict_data['Faculty']},
        6. Gender: {dict_data['Gender']},
        7. Citizenship: {dict_data['Citizenship']},
        8. Type of entrance exam: {dict_data['Type of entrance exam']}
        9. Exam subject 1: {dict_data['Exam subject 1']}
        10. Exam subject 2: {dict_data['Exam subject 2']}
        11. Exam subject 3: {dict_data['Exam subject 3']}
        12. Score on exam 1: {dict_data['Score on exam 1']}
        13. Score on exam 2: {dict_data['Score on exam 2']}
        14. Score on exam 3: {dict_data['Score on exam 3']}
        15. Type of olympiad: {dict_data['Type of olympiad']}
        16. Number of region: {dict_data['Number of region']},
        17. Kurs: {dict_data['Kurs']},
        18. Your study status: {dict_data['Your study status']}
        19. Training period in days: {dict_data['Training period in days']}
        """)

    # evaluate and return prediction text
    button = st.button('Predict')
    if button:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output[0]}")
        st.success('Success!')
