from tkinter import *
import tkinter as tk
import pandas as pd
import nltk
import numpy as np
import random
import csv
from datetime import datetime
import pendulum
from tkinter import messagebox

# Feature Extraction Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

import warnings

warnings.filterwarnings("ignore")

from test import Preprocessing, FeatureExtraction, Models

BG_GRAY = "#ABB2B9"
BG_COLOR = "#797EF6"
TEXT_COLOR_BLACK = "#000000"
TEXT_COLOR_WHITE = "#FFFFFF"
BUTTON_COLOR = "#34207E"
RADIO_BUTTON_COLOR = "#488AC7"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

dataset_folder = 'Datasets/'
saved_model_folder = 'SavedModels/'
desktop_path = 'C:/Users/Sami/Documents/Thesis'


class Emotion:
    def __init__(self):
        self.Emotions = df_emotion['sentiment'].unique()
        accuracies = np.array([svm_summary_ed['Accuracy'], lr_summary_ed['Accuracy'], rfc_summary_ed['Accuracy'],
                               mnb_summary_ed['Accuracy'], dt_summary_ed['Accuracy'], mlp_summary_ed['Accuracy']])
        norm_accuracy = accuracies - min(accuracies)
        self.emotion_model_weight = norm_accuracy / sum(norm_accuracy)

    def extract_best_emotion(self, list_emotion_pred):
        emotion_scores = {}
        for emotions in self.Emotions:
            emotion_scores[emotions] = 0.0
        for i in range(len(list_emotion_pred)):
            emotion_scores[list_emotion_pred[i]] += self.emotion_model_weight[i]
        se = sorted(emotion_scores.items(), key=lambda pair: pair[1], reverse=True)
        return se[0][0], round(se[0][1], 2)

    def detect_emotion(self, text):
        processed_text = fe_ed.get_processed_text(text)

        svm_emotion = svm_ed.predict(processed_text)[0]
        lr_emotion = logisticRegr_ed.predict(processed_text)[0]
        dt_emotion = dt_ed.predict(processed_text)[0]
        mnb_emotion = mnb_ed.predict(processed_text)[0]
        # xgbc_emotion = xgbc_ed.predict(processed_text)[0]
        rfc_emotion = rfc_ed.predict(processed_text)[0]
        mlp_emotion = mlp_ed.predict(processed_text)[0]

        list_emotion_pred = [svm_emotion, lr_emotion, rfc_emotion, mnb_emotion, dt_emotion, mlp_emotion]
        best_emotion, prob = self.extract_best_emotion(list_emotion_pred)
        print('Best Emotion:', best_emotion, ':', prob)

        print('Emotion using SVM: ', end='')
        print(svm_emotion)
        print('Emotion using Logistic Regression: ', end='')
        print(lr_emotion)
        print('Emotion using Decision Tree: ', end='')
        print(dt_emotion)
        print('Emotion using Naive Bayes: ', end='')
        print(mnb_emotion)
        print('Emotion using Random Forest: ', end='')
        print(rfc_emotion)
        print('Emotion using Multi-Layer Perceptron ', end='')
        print(mlp_emotion)
        print()
        return best_emotion, prob


class Application:

    def __init__(self):
        self.window = Tk()
        self.emo1, self.emo2, self.emo3, self.emo4, self.emo5, self.emo6, self.emo7 = None, None, None, None, None, None, None
        self._setup_main_window()
        self.emotion = Emotion()
        self.msg_count = 0
        self.user_msg = ''  # user message
        self.best_intent = ''  # best intent after applying ensemble model
        self.emo_prob = 0.0  # emotion probability predicted by emotion detection model

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit? All chats will be deleted"):
            # self.store_feedback()
            self.window.destroy()

    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=700, height=630, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR_WHITE,
                           text="Emotion Detection", font=(FONT_BOLD, 16), pady=12)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg="#002366")
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=21, bg="#8CD3FF", fg=TEXT_COLOR_BLACK,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.6, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg="#FFC0CB", height=2)
        bottom_label.place(relwidth=1, rely=0.68, relheight=0.35)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#FFFFFF", fg=TEXT_COLOR_BLACK, font=FONT)
        self.msg_entry.place(relwidth=0.70, relheight=0.15, rely=0.008, relx=0.01)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(bottom_label, text="Send", font=TEXT_COLOR_WHITE, width=15, fg=TEXT_COLOR_WHITE,
                             bg=BUTTON_COLOR,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.715, rely=0.008, relheight=0.15, relwidth=0.27)

        # Emotion Label
        self.emotion = Label(bottom_label, bg=BG_COLOR, justify=tk.LEFT, fg=TEXT_COLOR_WHITE, font=FONT_BOLD,
                             text="Emotion : ")
        self.emotion.place(relwidth=0.30, relheight=0.15, rely=0.183, relx=0.01)

        # emotion widget label
        self.emotion_widget = Text(bottom_label, width=30, height=3, bg=BG_COLOR, fg=TEXT_COLOR_WHITE,
                                   font="Helvetica 15 bold italic", padx=190, pady=5)
        self.emotion_widget.place(relheight=0.15, relwidth=0.67, rely=0.18, relx=0.315)
        self.emotion_widget.configure(cursor="arrow", state=DISABLED)


    def _on_enter_pressed(self, event):
        self.user_msg = self.msg_entry.get()
        self.msg_count += 1
        self._insert_message(self.user_msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        # # self.reply, self.intent_prob, self.best_intent = self.chatbot.chatbot_reply(msg)
        self.chat_emotion, self.emo_prob = self.emotion.detect_emotion(msg)

        self.emotion_widget.delete('1.0', END)
        self.emotion_widget.configure(state=NORMAL)
        self.emotion_widget.insert(END, self.chat_emotion.strip().capitalize())
        self.is_active_vote = True

        self.text_widget.see(END)


if __name__ == "__main__":


    df_emotion = pd.read_csv(dataset_folder + 'text_emotions_neutral.csv')

    # Train Test Split
    X_train_ed, X_test_ed, y_train_ed, y_test_ed = train_test_split(df_emotion['content'], df_emotion['sentiment'],
                                                                    test_size=0.3, random_state=116)


    fe_ed = FeatureExtraction(rmv_stopword=True)

    x_train_ed, x_test_ed = fe_ed.get_features(X_train_ed, X_test_ed)
    # Load Models
    emotion_models = Models(x_train_ed, y_train_ed, x_test_ed, y_test_ed, model_name='ed')

    svm_ed, logisticRegr_ed, rfc_ed, mnb_ed, dt_ed, mlp_ed = emotion_models.load_models()
    svm_summary_ed, lr_summary_ed, rfc_summary_ed, mnb_summary_ed, dt_summary_ed, mlp_summary_ed = emotion_models.model_summary()

    # svm_ed, logisticRegr_ed, rfc_ed, mnb_ed, dt_ed, mlp_ed = emotion_models.train_models()
    # svm_summary_ed, lr_summary_ed, rfc_summary_ed,  mnb_summary_ed, dt_summary_ed, mlp_summary_ed = emotion_models.model_summary()
    # emotion_models.save_models()

    app = Application()
    app.run()
