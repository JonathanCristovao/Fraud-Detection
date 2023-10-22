import pandas as pd
import streamlit as st
import pickle

st.subheader("Prevendo Risco de possivel fraude para o cartão de crédito")

model = pickle.load(open("model.pickle", "rb"))


def get_user_input():
    ProductCD = st.sidebar.slider("ProductCD", 0, 5, 1)
    TransactionDT = st.sidebar.slider("TransactionDT", 0, 1000000, 3000)
    options = {"gmail.com": 1, "yahoo.com": 2, "hotmail.com": 3}
    selected_email = st.selectbox(
        "Selecione um domínio de e-mail:", list(options.keys())
    )
    P_emaildomain = options[selected_email]
    TransactionAmt = st.sidebar.slider("TransactionAmt", 0, 10000, 3000)
    card1 = st.sidebar.slider("card1", 0, 10000, 3000)
    card2 = st.sidebar.slider("card2", 0, 600, 2)

    v = st.checkbox("Visa")
    m = st.checkbox("MasterCard")
    a = st.checkbox("American Express")
    d = st.checkbox("Discover")

    if v:
        card4 = 0
    elif m:
        card4 = 1
    elif a:
        card4 = 2
    elif d:
        card4 = 3
    else:
        card4 = 5
    card5 = st.sidebar.slider("card5", 0, 600, 2)
    addr1 = st.sidebar.slider("addr1", 0, 600, 2)
    D1 = st.sidebar.slider("D1", 0, 600, 2)
    D10 = st.sidebar.slider("D10", 0, 600, 2)
    D15 = st.sidebar.slider("D15", 0, 600, 2)
    V310 = st.sidebar.slider("V310", 0, 10000, 3000)
    V312 = st.sidebar.slider("V312", 0, 10000, 3000)
    V315 = st.sidebar.slider("V315", 0, 10000, 3000)
    V318 = st.sidebar.slider("V318", 0, 10000, 3000)
    V285 = st.sidebar.slider("V285", 0, 10000, 3000)
    V314 = st.sidebar.slider("V314", 0, 10000, 3000)
    V313 = st.sidebar.slider("V313", 0, 10000, 3000)
    V283 = st.sidebar.slider("V283", 0, 10000, 3000)
    V130 = st.sidebar.slider("V130", 0, 10000, 3000)
    V320 = st.sidebar.slider("V320", 0, 10000, 3000)

    user_data = {
        "card1": card1,
        "TransactionDT": TransactionDT,
        "addr1": addr1,
        "TransactionAmt": TransactionAmt,
        "D10": D10,
        "V310": V310,
        "card2": card2,
        "P_emaildomain": P_emaildomain,
        "D15": D15,
        "D1": D1,
        "V312": V312,
        "V315": V315,
        "V318": V318,
        "V285": V285,
        "card4": card4,
        "V314": V314,
        "card5": card5,
        "V313": V313,
        "V283": V283,
        "V130": V130,
        "V320": V320,
        "ProductCD": ProductCD,
    }
    features = pd.DataFrame(user_data, index=[0])

    return features


prediction = model.predict(get_user_input().values)
emoji_size = "50px" 
if prediction >= 0.7:
    st.subheader("Alta probabilidade dessa conta se ilícita!")
    emoji ="🤯"
elif prediction >=0.45 and prediction < 0.7:
    st.subheader("Média probabilidade dessa conta se ilícita!")
    emoji ="🤔"
else:
    st.subheader("Baixa probabilidade dessa conta se ilícita!")
    emoji ="🙂"

st.write(f'<div style="font-size:{emoji_size}">{emoji}</div>', unsafe_allow_html=True)   
porcentagem = round(prediction[0] * 100, 2)
st.subheader('Probabilidade: {:.2f}%'.format(porcentagem))
