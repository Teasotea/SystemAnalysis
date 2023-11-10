import time

import streamlit as st

# TODO: create fundtion that will return solution


def getSolution(params, pbar_container=st, max_deg=15):
    """
    params: dict with keys:
        "dimensions": [x1_dim, x2_dim, x3_dim, y_dim], # Розмірності векторів
        "input_file": input_file_text, # Текст вхідного файлу
        "output_file": output_file + ".xlsx", # Назва вихідного файлу
        "degrees": [x1_deg, x2_deg, x3_deg], # Степені поліномів (введіть нульові для перебору та пошуку найкращих)
        "weights": weight_method,  # "Ваги цільових функцій", ["Нормоване значення", "Середнє арифметичне"]
        "poly_type": poly_type, # "Вигляд поліномів", ["Чебишова", "Лежандра", "Лаґерра", "Ерміта"]
        "lambda_multiblock": lambda_option, # Визначати λ з трьох систем рівнянь

    """
    time.sleep(2)
    pass


st.set_page_config(
    page_title="Lab 2 System Analysis",
    page_icon="📈",
    layout="wide",  # "centered"
    menu_items={
        "About": "Лабораторна робота №2 з системного аналізу. Виконали: Шапошнікова Софія (КА-02) та Шушар Антон (КА-01)"
    },
)

page_bg_color = "#ccffcc"
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {page_bg_color};
    }}
    .centered-text {{
        text-align: center;
    }}
    .centered-button {{
        display: block;
        margin: 0 auto;
        width: 50%;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(
    "Відтворення функціональних залежностей у задачах розкриття концептуальної невизначеності"
)
st.write("Виконали: Шапошнікова Софія (КА-02) та Шушар Антон (КА-01)")

st.markdown("<h1 class='centered-text'>Поліноми</h1>", unsafe_allow_html=True)
poly_type = st.radio("Вигляд поліномів", ["Чебишова", "Лежандра", "Лаґерра", "Ерміта"])
st.write("Степені поліномів (введіть нульові для перебору та пошуку найкращих)")
cols = st.columns(3)
x1_deg = cols[0].number_input("для X1", value=13, step=1, key="x1_deg")
x2_deg = cols[1].number_input("для X2", value=11, step=1, key="x2_deg")
x3_deg = cols[2].number_input("для X3", value=7, step=1, key="x3_deg")
st.markdown("---")

st.markdown("<h1 class='centered-text'>Вектори</h1>", unsafe_allow_html=True)
cols = st.columns(4)
x1_dim = cols[0].number_input("Розмірність X1", value=2, step=1, key="x1_dim")
x2_dim = cols[1].number_input("Розмірність X2", value=2, step=1, key="x2_dim")
x3_dim = cols[2].number_input("Розмірність X3", value=3, step=1, key="x3_dim")
y_dim = cols[3].number_input("Розмірність Y", value=4, step=1, key="y_dim")
st.markdown("---")

st.markdown("<h1 class='centered-text'>Додатково</h1>", unsafe_allow_html=True)
weight_method = st.radio(
    "Ваги цільових функцій", ["Нормоване значення", "Середнє арифметичне"]
)
lambda_option = st.checkbox("Визначати λ з трьох систем рівнянь")
normed_plots = st.checkbox("Графіки для нормованих значень")

with st.sidebar:
    st.header("Дані")
    input_file = st.file_uploader(
        "Файл вхідних даних", type=["csv", "txt"], key="input_file"
    )
    output_file = st.text_input(
        "Назва файлу вихідних даних", value="output", key="output_file"
    )
    col_sep = st.selectbox(
        "Розділювач колонок даних",
        ("символ табуляції (типове значення)", "пробіл", "кома"),
        key="col_sep",
    )
    dec_sep = st.selectbox(
        "Розділювач дробової частини",
        ("крапка (типове значення)", "кома"),
        key="dec_sep",
    )

st.markdown('<div class="centered-button-container">', unsafe_allow_html=True)
if st.button("ВИКОНАТИ", key="run"):
    st.markdown("</div>", unsafe_allow_html=True)
    if input_file is None:
        st.error("**Помилка:** виберіть файл вхідних даних")
    elif x1_deg < 0 or x2_deg < 0 or x3_deg < 0:
        st.error("**Помилка:** степені поліномів не можуть бути від'ємними.")
    elif dec_sep == "кома" and col_sep == "кома":
        st.error(
            "**Помилка:** кома не може бути одночасно розділювачем колонок та дробової частини"
        )
    else:
        input_file_text = input_file.getvalue().decode()
        if dec_sep == "кома":
            input_file_text = input_file_text.replace(",", ".")
        if col_sep == "пробіл":
            input_file_text = input_file_text.replace(" ", "\t")
        elif col_sep == "кома":
            input_file_text = input_file_text.replace(",", "\t")
        params = {
            "dimensions": [x1_dim, x2_dim, x3_dim, y_dim],
            "input_file": input_file_text,
            "output_file": output_file + ".xlsx",
            "degrees": [x1_deg, x2_deg, x3_deg],
            "weights": weight_method,
            "poly_type": poly_type,
            "lambda_multiblock": lambda_option,
        }
        with st.spinner("Зачекайте..."):
            results = getSolution(params, pbar_container=st, max_deg=15)