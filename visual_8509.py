"""Visual system loaded exclusively by app_8509.py."""
from html import escape
import streamlit as st


def theme():
    st.markdown('''<style>
    .stApp {background:linear-gradient(145deg,#f5f8fc 0%,#edf5f3 100%); color:#19352f;}
    html,body,.stApp {font-family:Arial,Helvetica,sans-serif;}
    [data-testid="stSidebar"] {background:#103c35;} [data-testid="stSidebar"] * {color:#eaf5f1;}
    [data-testid="stSidebar"] button p {color:#19352f;}
    .block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem;}
    h1,h2,h3 {letter-spacing:-.035em;color:#173f35;font-family:Arial,sans-serif;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:20px;background:#ffffff;box-shadow:0 5px 20px #103c3508;}
    [data-testid="stMetric"] {padding:14px 18px;border-radius:16px;background:#e8f3ef;}
    button[kind="primary"] {background:#126552;border:0;border-radius:12px;min-height:42px;}
    .stButton>button,.stDownloadButton>button {border-radius:11px;font-weight:600;}
    .stTextInput input,.stNumberInput input {border-radius:10px;}
    [data-testid="stImage"] img {border-radius:14px;object-fit:contain;}
    [data-testid="stAppDeployButton"] {display:none;}
    [class*="st-key-product_card_"] {background:white;border-radius:18px;border-color:#dce8e2;box-shadow:0 4px 14px #123e3507;}
    [class*="st-key-product_card_"] h4 {font-size:1.15rem;line-height:1.4;letter-spacing:-.02em;min-height:3.2rem;}
    [class*="st-key-product_card_"] [data-testid="stImage"] img {height:180px;width:100%;background:#f5f8f6;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
    .ws-hero {padding:24px 30px;border-radius:24px;background:#123e35;color:white;margin-bottom:22px;}
    .ws-hero h1 {color:white;margin:0;font-size:2.3rem;}.ws-hero p {color:#c9e5db;margin:8px 0 0;}
    .ws-pill {display:inline-block;padding:4px 10px;border-radius:30px;background:#e8f3ef;color:#17634f;font-size:12px;}
    .ws-overline {font-size:.7rem;letter-spacing:.17em;font-weight:700;color:#a8d7c6;margin-bottom:12px;}
    .ws-photo-placeholder {height:180px;border:1px dashed #c5d9cd;border-radius:14px;background:linear-gradient(140deg,#f5faf7,#e5f0e9);display:flex;flex-direction:column;align-items:center;justify-content:center;color:#416b58;}
    .ws-photo-placeholder strong {font-size:2.8rem;letter-spacing:-.06em;}.ws-photo-placeholder small {font-size:.7rem;letter-spacing:.12em;margin-top:8px;}
    .ws-steps {display:flex;gap:8px;margin:16px 0;}.ws-step {flex:1;border-radius:10px;padding:12px 14px;background:#e7eee9;color:#60766b;font-size:.85rem;}.ws-step.active {background:#126552;color:white;font-weight:700;}
    @media(max-width:760px){.block-container{padding:1rem}.ws-hero{padding:18px}.ws-hero h1{font-size:1.7rem}}
    </style>''',unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="ws-hero"><div class="ws-overline">WEB&amp;SHOP / INVENTORYLENS</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',unsafe_allow_html=True)


def photo_placeholder(name):
    initials=''.join(word[0] for word in name.split()[:2]).upper()
    st.markdown(f'<div class="ws-photo-placeholder" aria-label="Reference photos required"><strong>{escape(initials)}</strong><small>REFERENCE PHOTOS REQUIRED</small></div>',unsafe_allow_html=True)


def registration_steps(index):
    labels=['Product details','Reference photos','Review & register']
    content=''.join(f'<div class="ws-step{" active" if i==min(index,2) else ""}">{i+1}. {label}</div>' for i,label in enumerate(labels))
    st.markdown(f'<div class="ws-steps" aria-label="Registration steps">{content}</div>',unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _thumbnail(path, modified_ns):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((320,320))
        return image.copy()


def thumbnail(path):
    from pathlib import Path
    return _thumbnail(str(path),Path(path).stat().st_mtime_ns)
