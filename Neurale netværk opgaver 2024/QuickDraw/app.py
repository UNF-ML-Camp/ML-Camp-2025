"""En simpel app til at tegne og gætte hvad der er tegnet"""

import os
import time
from datetime import datetime
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import cv2
import numpy as np
from data._data_static import TEGNINGER, dsize
import pandas as pd

@st.cache_resource
def get_model(path):
    """Læs modellen fra den gemte fil"""
    model = torch.jit.load(path)

    # Sæt modellen til evaluation mode
    model.eval()
    return model

def add_message(test_data: list[dict]):
    """Tilføj en besked til chat historikken"""
    # Præsenter modellernes gæt
    st.session_state.messages.append({
        "type": "assistant",
        "subtype": "prediction",
        "message": "Her er modellernes gæt:",
        "værdier": test_data,
    })

def predict(models: dict[str, torch.jit.ScriptModule], tegning: np.ndarray):
    """Få modellen til at forudsige hvad der er tegnet"""
    # Forbered tegningen til klassificering
    if tegning is not None:
        tegning_til_klassificering = cv2.cvtColor(tegning.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        tegning_til_klassificering = cv2.bitwise_not(tegning_til_klassificering)
        tegning_til_klassificering = cv2.resize(tegning_til_klassificering, dsize = dsize, interpolation = cv2.INTER_AREA)

        test_data = []
        # Forudsig hvad der er tegnet, hvis der er en tegning
        if tegning_til_klassificering.max() > 0:
            for model_name, model in models.items():
                start_time = time.time()
                image = torch.tensor(tegning_til_klassificering).float()
                image = image.to(next(model.parameters()).device)
                #image = image.reshape(-1, 1, 28, 28)
                
                with torch.no_grad():
                    y_hat_prob = model(image)
                    y_hat_prob = torch.nn.functional.softmax(y_hat_prob, dim=1)
                    y_hat = torch.argmax(y_hat_prob, dim=1)

                y_hat = y_hat.detach().numpy()[0]
                y_hat_prob = y_hat_prob[0].detach().numpy()

                prob = round(y_hat_prob[y_hat] * 100, 2)
                execution_time = time.time() - start_time

                test_data.append({
                    "model": model_name,
                    "pred_y": TEGNINGER[y_hat],
                    "pred_prob": prob,
                    "execution_time": execution_time,
                })

            # Returner forudsigelsen i appen
            test_data = pd.DataFrame.from_records(test_data)

            if len(test_data) == len(st.session_state.models) and len(test_data) != 0: # hvis der er en prediction for hver model
                if len(st.session_state.messages) == 0: # hvis der ikke er nogen beskeder
                    add_message(test_data)
                elif (
                  (st.session_state.messages[-1]["værdier"]["pred_prob"] != test_data["pred_prob"]).all() or
                  (st.session_state.messages[-1]["værdier"]["pred_y"] != test_data["pred_y"]).all()
                ): # hvis vi ikke allerede har inkluderet denne prediction
                    add_message(test_data)

@st.experimental_fragment()
def write_tegn_og_gaet(models: dict[str, torch.jit.ScriptModule]):
    # To colonner: Første til tegning, anden til api forbindelser
    col1, _, col2 = st.columns([1, 0.05, 1])

    # Tegneboks
    with col1:
        streg = st.slider("Stregtykkelse", 25, 50, 30, 1)
        tegning = st_canvas(
            fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
            stroke_width= streg,
            stroke_color="#000000",
            background_color="#FFFFFF",
            background_image = None,
            update_streamlit=True,
            height = 750,
            width = 750,
            drawing_mode="freedraw",
            key="canvas",
        )
        
    with col2:
        st.write("")
        st.write("")

        # Prædikter hvad der er tegnet, hvis der er en tegning
        if tegning != None and tegning.image_data is not None:
            if ((tegning.image_data != 255).any()):
                predict(models, tegning.image_data)

        # Vis chat historik
        with st.container(height=800, border=True):
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["type"]):
                    st.markdown(message["message"])
                    message_df = pd.DataFrame(message["værdier"])
                    st.dataframe(message_df)
                    y_true = st.selectbox("Hvad er det rigtige label?", TEGNINGER, index=None, key=f"dropdown_{i}")
                    message_df["y_true"] = y_true
                    st.session_state.testing = pd.concat([st.session_state.testing, message_df])

    # Hvis test historikken ikke er tom, vis den
    st.divider()
    if len(st.session_state.testing) > 0:
        st.subheader("Test session:")
        df_testing = st.session_state.testing
        df_testing = df_testing.dropna()
        df_testing = df_testing.drop_duplicates()
        df_testing.reset_index(drop=True, inplace=True)
        st.dataframe(df_testing)

        _, col3 = st.columns([1, 0.2])
        save_button = col3.button("Gem test sessionen", type="primary")

        if save_button:
            df_testing.to_csv(f"test_sessions/test_session_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv", index=False)

# Indstillinger
st.set_page_config(page_title="UNFML24 Tegn og Gæt", page_icon=".streamlit/unflogo.svg", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize testing history
if "testing" not in st.session_state:
    st.session_state.testing = pd.DataFrame()

# Initialize models to test
if "models" not in st.session_state:
    st.session_state.models = dict()

# Title på side
col1, _, col2 = st.columns([10, 1, 2])
col1.title("Tegn og gæt konkurrence")
col1.subheader(f"Hvilke modeller vil du gerne teste mod hinanden?")
with st.form(key='model_choice'):
    included_models = {
        model[:-4]: col1.checkbox(model[:-4], value=True)
        for model in os.listdir("saved_models")
    }
    submit_button = st.form_submit_button(
        label='Vælg modeller',
        type = "primary",
        use_container_width=True,
    )

    reload_button = st.form_submit_button(
        label='Genload modeller og nulstil chat',
        type = "secondary",
        use_container_width=True,
    )

    if submit_button:
        st.subheader(f"Valgte modeler:")
        for model, included in included_models.items():
            if included:
                st.session_state.models[model] = get_model(f"saved_models/{model}.pth")
                st.markdown(f"- {model}")
    
    if reload_button:
        st.session_state.models = {}
        st.cache_resource.clear()
        st.session_state.messages = []
        st.session_state.testing = pd.DataFrame()

col2.image('.streamlit/unflogo.svg', width=160)

st.divider()

write_tegn_og_gaet(models = st.session_state.models)

st.caption("UNFML24 Tegn og Gæt konkurrence - bygget af UNFML24 with :heart:")