import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu


# Set judul halaman
st.set_page_config(page_title="Prediksi Tingkat Kekerasan Anak", page_icon="📊", layout="wide")

# Fungsi untuk menyoroti kolom atribut dan label
def highlight_columns(x):
    attr_style = 'background-color:'
    label_style = 'background-color: yellow;'
    
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    df.iloc[:, :-1] = attr_style
    df.iloc[:, -1] = label_style
    return df

# Fungsi untuk mengisi nilai yang hilang
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == object:
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df

# Navigasi sidebar dengan ikon
with st.sidebar:
    st.sidebar.title("Navigasi")
    menu = option_menu(
        menu_title=None,
        options=["Dashboard", "C45"],
        icons=["house", "bar-chart"],
        menu_icon="cast",
        default_index=0
    )

# Inisialisasi state untuk dataset
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clf' not in st.session_state:
    st.session_state.clf = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'year_range' not in st.session_state:
    st.session_state.year_range = None

if menu == "Dashboard":
    st.title("Aplikasi Klasifikasi Tingkat Kekerasan Anak di Jawa Barat Menggunakan Metode C45")
    st.markdown(
        """
        <div class="centered">
            <p>Silahkan pilih menu di samping untuk mulai.</p>
            <p>Dengan adanya teknologi data mining, dapat menghasilkan data yang akurat dan memungkinkan tindakan yang lebih cepat dan tepat dalam menangani kasus kekerasan anak. 
            Implementasi teknologi data mining dan klasifikasi daerah rentan kekerasan anak diharapkan tidak hanya membantu dalam mengidentifikasi pola-pola kekerasan, 
            tetapi juga dalam memberikan respons yang lebih cepat dan tepat, sehingga upaya pencegahan dan penanganan kekerasan anak dapat dilakukan secara lebih efektif dan efisien..</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "C45":
    submenu = option_menu(
        menu_title=None,
        options=["Dataset", "Atribut Label", "Klasifikasi", "Prediksi"],
        icons=["cloud-upload", "tags", "graph-up-arrow", "box-arrow-in-right"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    if submenu == "Dataset":
        st.title("Dataset")
        st.write("Unggah dan kelola dataset Anda di sini.")
        
        # Unggah file
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        
        if uploaded_file is not None:
            # Baca file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Tampilkan jumlah data yang diunggah
            st.write(f"Jumlah data yang diunggah: {len(df)}")
            
            # Memungkinkan pengguna untuk memilih rentang tahun
            if 'Tahun' in df.columns:
                min_year = int(df['Tahun'].min())
                max_year = int(df['Tahun'].max())
                start_year = st.number_input("Pilih Tahun Mulai", min_value=min_year, max_value=max_year, value=min_year)
                end_year = st.number_input("Pilih Tahun Akhir", min_value=min_year, max_value=max_year, value=max_year)
                st.session_state.year_range = (start_year, end_year)
                df = df[(df['Tahun'] >= start_year) & (df['Tahun'] <= end_year)]
                st.session_state.df = df
                
                # Tampilkan jumlah data untuk rentang tahun yang dipilih
                st.write(f"Jumlah data untuk rentang tahun {start_year} - {end_year}: {len(df)}")
            
            # Tampilkan dataframe
            st.write("Dataset")
            st.dataframe(df)

    elif submenu == "Atribut Label":
        st.title("Atribut Label")
        st.write("Definisikan atribut dan label untuk dataset Anda.")
        
        if st.session_state.df is not None:
            # Hapus kolom yang tidak diperlukan untuk preprocessing
            df = st.session_state.df.drop(columns=['Provinsi','Tahun','Kabupaten_Kota','Kategori_Jumlah_Korban'], errors='ignore')
            st.session_state.df = fill_missing_values(df)

            # Tampilkan dataframe dengan kolom yang disorot
            st.write("Data yang ditampilkan berikut telah melalui tahap transformasi (penghapusan kolom, penyesuaian type data, dsb).Dataset atribut label adalah yang disorot")
            st.dataframe(df.style.apply(highlight_columns, axis=None))
        else:
            st.write("Silakan unggah dataset di submenu 'Dataset'.")

    elif submenu == "Klasifikasi":
        st.title("Klasifikasi")
        st.write("Evaluasi kinerja klasifikasi di sini.")
        
        if st.session_state.df is not None:
            # Pisahkan data menjadi atribut dan label
            df = st.session_state.df.copy()
            df = fill_missing_values(df)
            X = df.iloc[:, :-1]  # Semua kolom kecuali yang terakhir
            y = df.iloc[:, -1]   # Kolom terakhir
            
            # Encode variabel kategorikal
            label_encoders = {}
            for column in X.columns:
                if X[column].dtype == object:
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column].astype(str))
                    label_encoders[column] = le
            
            if y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                label_encoders[df.columns[-1]] = le

            st.session_state.label_encoders = label_encoders

            # Pilih persentase data untuk pelatihan dan pengujian
            train_size = st.number_input("Pilih persentase data untuk pelatihan", min_value=10, max_value=90, value=70)
            test_size = 100 - train_size
            
            # Pisahkan data menjadi set pelatihan dan pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, test_size=test_size/100, random_state=42)
            
            st.write(f"Data Pelatihan: {train_size}%")
            st.write(f"Data Pengujian: {test_size}%")

            # Latih model decision tree classifier
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            
            st.session_state.clf = clf
            
            # Prediksi
            y_pred = clf.predict(X_test)
            
            # Hitung akurasi
            accuracy = accuracy_score(y_test, y_pred)
            
            st.write(f"Akurasi: {accuracy * 100:.2f}%")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders[df.columns[-1]].classes_, yticklabels=label_encoders[df.columns[-1]].classes_)
            st.pyplot(fig)
            
            # Classification Report
            st.write("Laporan Klasifikasi")
            report = classification_report(y_test, y_pred, target_names=label_encoders[df.columns[-1]].classes_)
            st.text(report)
            
            # Plot decision tree sebagai visual
            st.write("Decision Tree yang digunakan (visual):")
            fig, ax = plt.subplots(figsize=(50, 25))
            plot_tree(clf, feature_names=X.columns, class_names=label_encoders[df.columns[-1]].classes_, filled=True, ax=ax)
            st.pyplot(fig)

        else:
            st.write("Silakan unggah dataset di submenu 'Dataset'.")

    elif submenu == "Prediksi":
        st.title("Prediksi")
        st.write("Lakukan prediksi berdasarkan model yang telah dilatih di sini.")
        
        if st.session_state.df is not None and st.session_state.clf is not None:
            df = st.session_state.df.copy()
            X = df.iloc[:, :-1]  # Semua kolom kecuali yang terakhir
            clf = st.session_state.clf
            label_encoders = st.session_state.label_encoders

            # Input prediksi manual
            st.write("Masukkan nilai untuk prediksi:")
            user_input = {}
            for column in X.columns:
                if X[column].dtype == object:
                    options = list(label_encoders[column].classes_)
                    user_input[column] = st.selectbox(column, options)
                else:
                    user_input[column] = st.number_input(column, value=0)

            if st.button("Prediksi"):
                input_df = pd.DataFrame([user_input])
                for column in input_df.columns:
                    if column in label_encoders:
                        input_df[column] = label_encoders[column].transform(input_df[column])
                
                prediction = clf.predict(input_df)[0]
                prediction_label = label_encoders[df.columns[-1]].inverse_transform([prediction])[0]

                # Tampilkan nilai inputan, hasil prediksi, dalam bentuk tabel
                st.write("Hasil Prediksi :")
                result_df = pd.DataFrame({
                    "Atribut": list(user_input.keys()) + ["Prediksi"],
                    "Nilai": list(user_input.values()) + [prediction_label]
                })
                st.table(result_df)
        else:
            st.write("Silakan unggah dataset di submenu 'Dataset' dan latih model di submenu 'Klasifikasi'.")