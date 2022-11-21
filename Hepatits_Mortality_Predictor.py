from managed_db import *
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import hashlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# DB
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    else:
        return False


feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites'
                      'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']
gender_dict = {"male": 1, 'female': 2}
feature_dict = {'No': 1, 'Yes': 2}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key


def get_feature_val(val):
    feature_dict = {'No': 1, 'Yes': 2}
    for key, value in feature_dict.items():
        if val == key:
            return value


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def main():
    """"Mortality Prediction App"""
    st.title("Desease Mortality Prediction App")

    menu = ['Home', 'Login', 'SignUp']
    submenu = ['Plot', 'Prediction']

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Home':
        st.subheader("Home")
        st.text('What is Hepatitis?')
        st.write('Hepatitis is an inflammation of the liver. The condition can be self-limiting or can progress to fibrosis (scarring), cirrhosis or liver cancer. Hepatitis viruses are the most common cause of hepatitis in the world but other infections, toxic substances (e.g. alcohol, certain drugs), and autoimmune diseases can also cause hepatitis.')
        st.write('There are 5 main hepatitis viruses, referred to as types A, B, C, D and E. These 5 types are of greatest concern because of the burden of illness and death they cause and the potential for outbreaks and epidemic spread. In particular, types B and C lead to chronic disease in hundreds of millions of people and, together, are the most common cause of liver cirrhosis and cancer.')
        st.write('Hepatitis A and E are typically caused by ingestion of contaminated food or water. Hepatitis B, C and D usually occur as a result of parenteral contact with infected body fluids. Common modes of transmission for these viruses include receipt of contaminated blood or blood products, invasive medical procedures using contaminated equipment and for hepatitis B transmission from mother to baby at birth, from family member to child, and also by sexual contact. ')
        st.write('Acute infection may occur with limited or no symptoms, or may include symptoms such as jaundice (yellowing of the skin and eyes), dark urine, extreme fatigue, nausea, vomiting and abdominal pain.')

    elif choice == 'Login':
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.checkbox('Login'):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username, verify_hashes(password, hashed_pswd))
            if result:
                st.success('Welcome {}'.format(username))

                activity = st.selectbox('Activity', submenu)
                if activity == 'Plot':
                    st.subheader('Data Visualisation Plot')
                    df = pd.read_csv('clean_hepatitis_dataset.csv')
                    st.dataframe(df)
                    fig, ax = plt.subplots()
                    ax = df['class'].value_counts().plot(kind='bar')
                    st.pyplot(fig)

                    freq_df = pd.read_csv('freq_df_hepatitis_dataset.csv')
                    st.bar_chart(freq_df['count'])

                    if st.checkbox('Area Chart'):
                        all_columns = df.columns.tolist()
                        feat_choices = st.multiselect(
                            'Choose a Feature', all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)

                elif activity == 'Prediction':
                    st.subheader('Predictive Analytics')
                    age = st.number_input('Age', 7, 80)
                    sex = st.radio('Sex', tuple(gender_dict.keys()))
                    steroid = st.radio('Do You Take Steroid?',
                                       tuple(feature_dict.keys()))
                    antivirals = st.radio('Do You Take Antivirals?',
                                          tuple(feature_dict.keys()))
                    fatigue = st.radio('Do Have Fatigue?',
                                       tuple(feature_dict.keys()))
                    spiders = st.radio('Presence of Spiders Naevi?',
                                       tuple(feature_dict.keys()))
                    ascites = st.selectbox('Ascites',
                                           tuple(feature_dict.keys()))
                    varices = st.selectbox(
                        'Presence of Varices', tuple(feature_dict.keys()))
                    bilirubin = st.number_input('Bilirubin Content', 0.0, 8.0)
                    alk_phosphate = st.number_input(
                        'Alkaline Phosphate Content', 0.0, 296.0)
                    sgot = st.number_input('Sgot', 0.0, 648.0)
                    albumin = st.number_input('Albumin', 0.0, 6.4)
                    protime = st.number_input('Prothrombin Time', 0.0, 100.0)
                    histology = st.selectbox(
                        'Histology', tuple(feature_dict.keys()))
                    feature_list = [age, get_value(sex, gender_dict), get_feature_val(
                        steroid), get_feature_val(antivirals), get_feature_val(fatigue), get_feature_val(spiders), get_feature_val(ascites), get_feature_val(varices), bilirubin, alk_phosphate, sgot, albumin, int(protime), get_feature_val(histology)]
                    st.write(len(feature_list))
                    st.write(feature_list)

                    pretty_result = {'age': age, 'sex': sex, 'steroid': steroid, 'antivirals': antivirals, 'fatigue': fatigue, 'spiders': spiders, 'ascites': ascites, 'varices': varices,
                                     'bilirubin': bilirubin, 'alk_phosphate': alk_phosphate, 'sgot': sgot, 'albumin': albumin, 'protime': protime, 'histology': histology}
                    st.json(pretty_result)
                    single_sample = np.array(feature_list).reshape(1, -1)
                    # st.write(single_sample)

                    # Model
                    model_choice = st.selectbox(
                        'Select Model', ['LogisticRegression', 'RandomForest', 'DecisionTree'])
                    if st.button('Predict'):
                        if model_choice == 'RandomForest':
                            loaded_model = load_model(
                                'models/RandomForestClassifier.pkl')
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(
                                single_sample)

                        elif model_choice == 'DecisionTree':
                            loaded_model = load_model(
                                'models/DecisionTree.pkl')
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(
                                single_sample)
                        else:
                            loaded_model = load_model(
                                'models/LR.pkl')

                        prediction = loaded_model.predict(single_sample)
                        pred_prob = loaded_model.predict_proba(
                            single_sample)

                        if prediction == 1:
                            st.warning('Patient Dies')
                            pred_probability_score = {
                                "Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}
                            st.subheader(
                                'Prediction Probability Score using {}'.format(model_choice))
                            st.json(pred_probability_score)
                        else:
                            st.success('Patient Lives')
                            pred_probability_score = {
                                "Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}
                            st.subheader(
                                'Prediction Probability Score using {}'.format(model_choice))
                            st.json(pred_probability_score)
                else:
                    st.warning('Incorrect Username/Password')

    elif choice == 'SignUp':
        new_username = st.text_input('User name')
        new_password = st.text_input('Password', type='password')
        confirm_password = st.text_input(
            'ConfirmPassword', type='password')
        if new_password == confirm_password:
            st.success('Password Confirmed')
        else:
            st.warning('Password not the same')

        if st.button('Submit'):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_usertable(new_username, hashed_new_password)
            st.success('You have successfuly created a new account')
            st.info('Login to Get Started')


if __name__ == '__main__':
    main()
