import streamlit as st

def build(medical_df):
    st.write("Welcome to the page. You can find out stuff about COVID-19 prediction here.")
    st.markdown("""
    We worked with publicly available COVID-19 data from New York during the first wave of the disease in early 2020.
    This data includes about 1400 __hosptialized__ patients including medical information in form of tabular data like _gender_, _age_, 
    _preconditions_ or _smoking-habits_ together with per-patient laboratory data like _blood pressure_. 

    Additionally the datasets contains medical image data in form of _Chest X-Ray AP_ images. This data is presented in the __Medical__ tab.
    and has been analysed with pyradiomics.

    more to come...
    """)

    st.write("Feel free to choose a tab above to explore stuff!")
    
    st.markdown("---")

    #st.write("New data: ", medical_df.shape)
    #st.dataframe(medical_df)