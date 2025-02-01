import streamlit as st
import pickle

# Fungsi untuk memuat model dan tokenizer dari file pickle
def load_model_from_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        model, tokenizer = pickle.load(file)
    return model, tokenizer

# Inisialisasi riwayat percakapan dan teks input menggunakan session state
if "conversation" not in st.session_state:
    st.session_state.conversation = [] 

if "user_input" not in st.session_state:
    st.session_state.user_input = "" 

# Fungsi untuk menghasilkan respons chatbot
def generate_response(input_text, pickle_path="model/chatbot_model.pkl"):
    # Muat model dan tokenizer dari file pickle
    try:
        model, tokenizer = load_model_from_pickle(pickle_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return "I'm sorry, I can't respond right now."

    # Gabungkan riwayat percakapan sebelumnya dengan input baru
    history_text = "".join(
        [f"<|user|> {msg['user']} <|bot|> {msg['bot']} " for msg in st.session_state.conversation]
    )
    full_context = f"<|startoftext|>{history_text}<|user|> {input_text} <|sep|>"

    # Encode input ke dalam token
    input_ids = tokenizer.encode(full_context, return_tensors="pt")

    # Generate output menggunakan model
    try:
        output = model.generate(
            input_ids,
            max_length=10000,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        st.error(f"Error during generation: {e}")
        return "I'm sorry, I can't generate a response right now."

    # Decode output menjadi teks
    response = tokenizer.decode(output[0], skip_special_tokens=False)

    # Hapus token spesial seperti <|endoftext|>
    response = response.replace("<|endoftext|>", "").strip()

    # Ambil hanya teks setelah <|sep|>, jika ada
    if "<|sep|>" in response:
        response = response.split('<|sep|>')[1].strip()

    return response

# Path ke file pickle model
pickle_model_path = "model/chatbot_model.pkl"

# Streamlit UI
st.title("Fanbot")
st.write("Selamat datang di Fanbot! Ketikkan sesuatu dan saya akan merespons. (gunakan bahasa inggris)")

# Form input pengguna
with st.form(key='user_input_form'):
    col1, col2 = st.columns([8, 1]) 
    with col1:
        user_input = st.text_input("You:",st.session_state.user_input)
    with col2:
        submit_button = st.form_submit_button(label='Send')
        
    st.markdown("""
        <style>
            .stFormSubmitButton {
                margin-top: 30px; 
            }
            @media screen and (max-width: 600px) {
                .stFormSubmitButton {
                    display: none;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    if submit_button:
        st.session_state.user_input = ""  

# Proses input dan respons
if submit_button and user_input:
    bot_response = generate_response(user_input, pickle_path=pickle_model_path)
    st.session_state.conversation.append({"user": user_input, "bot": bot_response})
    st.session_state.user_input = ""

# Menampilkan riwayat percakapan dalam bentuk bubble chat
for message in st.session_state.conversation:
    st.markdown(f"""
        <div style="background-color: #16291a; color: white; padding: 12px 15px; margin: 10px 0; border-radius: 15px; max-width: 80%; float: right; clear: both; display: block; font-size: 14px; text-align: right;">
          {message['user']}
        </div>
        <div style="background-color: #1d1e21; color: white; padding: 12px 15px; margin: 10px 0; border-radius: 15px; max-width: 80%; float: left; clear: both; display: block; font-size: 14px; text-align: left;">
          {message['bot']}
        </div>
    """, unsafe_allow_html=True)
