from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from graph import Graph
from model import Model


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
        "sample_size": sample_size, # Розмір вибірки
        ""

    """

    def get(*args):
        return [params.get(arg) for arg in args]

    additive = Model(
        input_file=params["input_file"],
        output_file=params["output_file"],
        sample_size=params["sample_size"],
        dimensions=params["dimensions"],
        degrees=params["degrees"],
        poly_type=params["poly_type"],
        lambda_multiblock=params["lambda_multiblock"],
    )

    additive.solve()

    result = []
    result.append(np.c_[['\\lambda'], [additive.l0]])
    result.append(np.c_[[f'a_{i+1}' for i in range(additive.ny)], np.hstack([additive.a1, additive.a2, additive.a3])])
    result.append(np.c_[[f'c_{i+1}' for i in range(additive.ny)], additive.c])
    result.extend(additive.print_phi())
    result.extend(additive.print_phi_extended())

    return (result, Graph(additive))


def print_array(array):
    def is_number(n):
        try:
            float(n)
        except ValueError:
            return False
        return True

    table = (
        rf"\begin{{array}}{{|{'|'.join(['c' for i, _ in enumerate(array)])}|}}\hline "
    )
    for e in np.array(array).T:
        table += (
            " & ".join([f"{float(d):.4f}" if is_number(d) else d for d in e])
            + r"\\ \hline"
        )

    table += rf"\end{{array}}"
    st.latex(table)


def print_latex(obj):
    if isinstance(obj, str):
        st.latex(obj)
    # elif isinstance(obj, list | np.ndarray):
    #     print_array(obj)
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        print_array(obj)


st.set_page_config(
    page_title="Lab 2 System Analysis",
    page_icon="📈",
    layout="wide",  # "centered"
    menu_items={
        "About": "Лабораторна робота №2 з системного аналізу. Виконали: Шапошнікова Софія (КА-02) та Шушар Антон (КА-01)"
    },
)

page_bg_color = "#ffffff"
st.markdown(
    f"""
    <style>
    .reportview-container, .appview-container {{
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

st.markdown("<h1 class='centered-text'>Вектори</h1>", unsafe_allow_html=True)
cols = st.columns(4)
x1_dim = cols[0].number_input("Розмірність X1", value=2, step=1, key="x1_dim")
x2_dim = cols[1].number_input("Розмірність X2", value=1, step=1, key="x2_dim")
x3_dim = cols[2].number_input("Розмірність X3", value=2, step=1, key="x3_dim")
y_dim = cols[3].number_input("Розмірність Y", value=3, step=1, key="y_dim")
st.markdown("---")

st.markdown("<h1 class='centered-text'>Поліноми</h1>", unsafe_allow_html=True)
poly_type = st.radio("Вигляд поліномів", ["Чебишова", "Лежандра", "Лаґерра", "Ерміта"])
st.write("Степені поліномів (введіть нульові для перебору та пошуку найкращих)")
cols = st.columns(3)
x1_deg = cols[0].number_input("для X1", value=6, step=1, key="x1_deg")
x2_deg = cols[1].number_input("для X2", value=5, step=1, key="x2_deg")
x3_deg = cols[2].number_input("для X3", value=1, step=1, key="x3_deg")
find_button = st.empty()
st.markdown("---")

st.markdown("<h1 class='centered-text'>Додатково</h1>", unsafe_allow_html=True)
weight_method = st.radio(
    "Ваги цільових функцій", ["Нормоване значення", "Середнє арифметичне"]
)
lambda_option = st.checkbox("Визначати λ з трьох систем рівнянь")
normed_plots = st.checkbox("Графіки для нормованих значень")

def get_params():
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
        file_text = StringIO(input_file_text)
        df = pd.read_csv(file_text, sep="\t", header=None)
        print(len(df.iloc[:, :-1].values.flatten()))

        return {
            "dimensions": tuple([x1_dim, x2_dim, x3_dim, y_dim]),
            "input_file": df.iloc[:, :-1].values.flatten(),
            "output_file": output_file + ".xlsx",
            "degrees": tuple([x1_deg, x2_deg, x3_deg]),
            "weights": weight_method,
            "poly_type": poly_type,
            "lambda_multiblock": lambda_option,
            "sample_size": len(df),
        }
    return None

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
        index=2,
    )
    dec_sep = st.selectbox(
        "Розділювач дробової частини",
        ("крапка (типове значення)", "кома"),
        key="dec_sep",
    )

def find_degrees():
    params = get_params()
    if params is not None:
        with st.spinner("Підбираються степені. Зачекайте..."):
            errors = {}
            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        model = Model(
                            input_file=params["input_file"],
                            output_file=params["output_file"],
                            sample_size=params["sample_size"],
                            dimensions=params["dimensions"],
                            degrees=(i, j, k),
                            poly_type=params["poly_type"],
                            lambda_multiblock=params["lambda_multiblock"],
                        )
                        model.solve()
                        errors.update({(i, j, k): model.error_normalized})
            st.session_state.x1_deg, st.session_state.x2_deg, st.session_state.x3_deg = min(errors, key=lambda k: errors[k].mean())


find_button.button("Підібрати степені", key="find_degrees", on_click=find_degrees)

st.markdown('<div class="centered-button-container">', unsafe_allow_html=True)
if st.button("ВИКОНАТИ", key="run"):
    st.markdown("</div>", unsafe_allow_html=True)
    params = get_params()
    if params is not None:
        with st.spinner("Зачекайте..."):
            results, graph = getSolution(params, pbar_container=st, max_deg=15)

            st.markdown(
                """
            <style>
            .katex-display>.katex {
                white-space: unset !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            for res in results:
                print_latex(res)

        st.subheader("Графіки")

        if normed_plots:
            st.pyplot(graph.plot_normalized())
        else:
            st.pyplot(graph.plot_predict())
