import urllib

from streamlit_option_menu import option_menu
import sklearn

from random import gauss
from math import floor

# Core Packages
import streamlit as st
import altair as alt

# EDA Packages
import pandas as pd
import numpy as np

# Utils
import joblib

import speech_recognition as sr


# load trained model
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_14_June_2022.pkl", "rb"))

# the emotion-to-task google sheet to use
sheet_url = "https://docs.google.com/spreadsheets/d/1hDa4-idBdmN2nVyZn2Oi6TmnYg-A8jFZFh1BXsEnYBA/edit#gid=0"
sheet_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
try:
    e2t_df = pd.read_csv(sheet_url)
except urllib.error.URLError as e:
    ResponseData = e.reason


# functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def affirmation_generator(max_emotion):
    if max_emotion == 'neutral':
        return "It seems like you are not having any strong emotions at the moment. Here are some tasks that you can \
        do while feeling this way 🍀"
    elif max_emotion == 'joy':
        return "We are happy you are happy :) \n\nTake this rush of positivity to the next level! \
        The tasks below encourage you to spread the positivity, document gratitude, and improve and reflect on yourself."
    elif max_emotion == 'sadness':
        return "It’s okay to not feel okay 🧸 \n\nSadness is a normal feeling, and it's important to let ourselves feel \
        this way. Once you feel ready, here are some tasks to help you relax, maintain connections with others, move \
        around, and self-reflect!\n\n Please seek help from a professional if your sadness becomes overwhelming or \
        feels unmanageable."
    elif max_emotion == 'anger':
        return "Something or someone might have frustrated you ⚡️\n\nHere are some suggested tasks to help you relax, \
        acknowledge and identify what has happened, look at what is in your control in this situation, and de-stress."
    elif max_emotion == 'fear':
        return "Here is a virtual hug 🌼\n\nFear and anxiety can stem from past, present, or future worries. Let’s \
        take some time to recognize the fear, connect with others, move your body, and relax.\n\nPlease seek help from \
        a professional if your fear becomes unmanageable or if you are in danger."
    elif max_emotion == 'disgust':
        return "The universal trigger for disgust is the feeling that something is offensive, poisonous, or \
        contaminating 🥀\n\nLet’s identify the trigger, see the good in others, and focus on self-improvement."
    elif max_emotion == 'surprise':
        return "Life throws unexpected things at us all the time 💫\n\nLet’s think positively, move around, and take \
        some time out. Then let’s see how the surprising elements affect your current and future plans."


def task_generator(prob_for_tasks, max_emotion, num_of_tasks):
    detected_emotion = [('neutral', prob_for_tasks[0][4]),
                        ('joy', prob_for_tasks[0][3]),
                        ('sadness', prob_for_tasks[0][5]),
                        ('anger', prob_for_tasks[0][0]),
                        ('fear', prob_for_tasks[0][2]),
                        ('disgust', prob_for_tasks[0][1]),
                        ('surprise', prob_for_tasks[0][7])]

    prob_dist = []  # probability distribution of the 7 emotions
    for i in range(7):
        prob_dist.append(detected_emotion[i][1])

    task_list_indices = []
    likes = e2t_df['Likes'].values
    max_likes_count = max(likes)
    min_likes_count = min(likes)

    for index, row in e2t_df[e2t_df[max_emotion] == 1].iterrows():
        emotion_encoding = e2t_df.iloc[index, 1:8].values
        match_score = round(np.dot(prob_dist, emotion_encoding), 4)
        normalized_likes = 5 * (row['Likes'] - min_likes_count) / (max_likes_count - min_likes_count)  # scale 0 to 5

        # rec rule: based on match score (scale by 100), popularity level (likes and rating),
        # and feasibility (easiness of the task == negative of points)
        # rule is adjustable
        rec_score = match_score * 100 + normalized_likes + row['Rating'] - row['Points']

        task_list_indices.append((index, rec_score))

    task_list_indices = sorted(task_list_indices, key=lambda x: x[1])

    rec_tasks = []
    for i in range(10):  # the value 10 can change
        task_index = task_list_indices.pop(-1)[0]
        rec_tasks.append(e2t_df.iat[task_index, 0])
        i += 1

    # throw in some randomness
    # determining the values of the parameters
    mu = 0
    sigma = 5

    # using the gauss() method
    final_list_indices = []
    size = 0
    while size != num_of_tasks:
        value = gauss(mu, sigma)
        abs_floor_value = floor(abs(value))
        if abs_floor_value not in final_list_indices and abs_floor_value <= 9:
            final_list_indices.append(abs_floor_value)
            size += 1

    final_task_list = []
    for index in final_list_indices:
        final_task_list.append(rec_tasks[index])

    return final_task_list


def result(raw_text):
    col1, col2 = st.columns(2)
    # applying functions
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)
    with col1:
        # st.success("Speech-to-Text")
        st.subheader("Your input:")
        st.write(raw_text)
        # st.success("Prediction")
        st.subheader("We think you are feeling...")
        emoji_icon = emotions_emoji_dict[prediction]
        st.write("{}:{}".format(prediction, emoji_icon))
        st.write("Emotion Percentage %: {}".format(round(np.max(probability*100),1)))

    with col2:
        # st.success("Prediction Probabilities")
        st.subheader("Emotion Prediction Probabilities:")
        # st.write(probability)
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        # st.write(proba_df.T)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]
        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        st.altair_chart(fig, use_container_width=True)

    st.subheader("We would like to tell you something...")
    st.write(affirmation_generator(prediction))

    st.subheader("Task list:")
    task_list = task_generator(probability, prediction, 5)
    for task in task_list:
        st.checkbox(task)


emotions_emoji_dict = {"anger": "😠", "disgust": "🤢", "fear": "😱", "happy": "😊", "joy": "😁", "neutral": "😐",
                       "sadness": "😞", "shame": "😳", "surprise": "😲"}


def main():

    menu = ["Our App", "About Us"]
    # choice_side = st.sidebar.selectbox("Menu", menu)

    choice_horizontal = option_menu(None, menu,
                         icons=['phone', 'book'],
                         menu_icon="cast", default_index=0, orientation="horizontal",
                         key='test')

    st.title("💭 emotiive mobile application")

    if choice_horizontal == "Our App":

        # TEXT

        st.subheader("Type how you're feeling today")
        with st.form(key='emotion_form_1'):

            raw_text = st.text_area("Type Here")
            submit = st.form_submit_button(label='submit')
            if submit:
                result(raw_text)

        #  AUDIO
        #  Nested buttons

        st.subheader("Or record how you're feeling")
        with st.form(key='emotion_form_2'):

            record = st.form_submit_button(label='🎙️')

            while record:

                r = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = r.listen(source)
                try:

                    raw_text = r.recognize_google(audio)
                    raw_text = st.text_area(label="Say Something", value=raw_text)
                    result(raw_text)
                    record = False


                # except UnknownValueError:

                # st.markdown("Cannot understand what you are saying!!")
                except sr.RequestError as e:
                    st.write("error; {0}".format(e))

                except Exception as e:
                    st.write(e)

    elif choice_horizontal == "About Us":
        st.subheader("About Us")
        st.markdown("*empowering users to better navigate their emotions and support their own mental health*")
        st.subheader("Our App")
        st.markdown("📱emotiive.ml app uses machine learning to identify user's emotions and generate psychology-based tasks to help users keep track of and process the emotions they experience throughout the day.")
        st.subheader("Team Behind emotiive.ml")
        st.markdown \
            ("✨ We're a group of women who are passionate about applying machine learning to the field of mental health and self-care. The team consists of an engineer-in-training, a graduate student, and three undergraduate students.")


if __name__ == '__main__':
    main()
