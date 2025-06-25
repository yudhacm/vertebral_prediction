import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('gnb_model.pkl')

# Judul aplikasi
st.title("Prediksi Kelainan Tulang Belakang 🦴")
st.markdown("Aplikasi ini memprediksi apakah MRI tulang belakang menunjukkan kondisi normal atau abnormal.")

# Pilih metode input
menu = st.radio("Pilih metode input:", ["Input Manual", "Upload CSV"])

if menu == "Input Manual":
    st.subheader("Masukkan Nilai Fitur:")
    pelvic_incidence = st.number_input("Pelvic Incidence")
    pelvic_tilt = st.number_input("Pelvic Tilt")
    lumbar_lordosis_angle = st.number_input("Lumbar Lordosis Angle")
    sacral_slope = st.number_input("Sacral Slope")
    pelvic_radius = st.number_input("Pelvic Radius")
    degree_spondylolisthesis = st.number_input("Degree Spondylolisthesis")

    if st.button("Prediksi"):
        df_input = pd.DataFrame([[pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle,
                                  sacral_slope, pelvic_radius, degree_spondylolisthesis]],
                                columns=['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                                         'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'])
        prediction = model.predict(df_input)[0]
        label = ['Abnormal', 'Normal', 'Abnormal (second)']
        st.success(f"Hasil Prediksi: **{label[prediction]}**")

elif menu == "Upload CSV":
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diupload:")
        st.dataframe(df)

        try:
            prediction = model.predict(df)
            label = ['Abnormal', 'Normal', 'Abnormal (second)']
            df['Prediksi'] = [label[i] for i in prediction]

            st.subheader("Hasil Prediksi")
            st.dataframe(df)

            # Download hasil
            csv = df.to_csv(index=False).encode()
            st.download_button("Download hasil prediksi", csv, "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error("Terjadi kesalahan saat prediksi: " + str(e))

st.markdown("---")
st.caption("Dibuat dengan Streamlit dan Gaussian Naive Bayes")
