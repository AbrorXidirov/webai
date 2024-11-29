import numpy as np
import streamlit as st
import pickle
st.title("Bemorni tekshirish:")

# Yosh
Age = st.number_input("Yosh: ", min_value=0, max_value=120)
# Glucose
Glukoza = st.number_input("Glukoza miqdori: ", format="%.1f", min_value=0.0)
# BMI vaznning bo'yga nisbati
BMI = st.number_input("BMI: ", format="%.1f", min_value=0.0)
# Pregnancies
Pregnancies = st.number_input("Pregnancies:", min_value=0)

# Tajribaga ega modelni chaqirish
try:
    with open('qarorlar_daraxti_model.pkl', 'rb') as file:
        decision_tree_model = pickle.load(file)
except Exception as e:
    st.error(f"Modelni yuklashda xato: {e}")

# Bashorat uchun tugma
if st.button("Bashorat qilish"):
    # Kiritilgan ma'lumotlarni massiv ko'rinishiga o'tkazish
    features = np.array([[Age, Glukoza, BMI, Pregnancies]])

    # Modelga kiritilgan ma'lumotlarni uzatamiz
    prediction = decision_tree_model.predict(features)

    # Natijani chiqaramiz
    if prediction[0] == 0:
        st.success("Bashorat: Sizda kasallik aniqlandi")
    else:
        st.success("Bashorat: Sizning holatingiz yaxshi, kasallik aniqlanmadi.")
