"""Isolated 8509 presentation and registration release; 8508 keeps app.py."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('WEBSHOP_DATA_DIR',str(ROOT/'data/webshop_ux_demo'))
if Path(os.environ['WEBSHOP_DATA_DIR']).resolve()==(ROOT/'data/webshop_release_demo').resolve():
    raise RuntimeError('Use the isolated 8509 data directory; the release demo is protected.')

import streamlit as st
import auth
import db
from inventory import InventoryService
from registration_8509 import init_registration
from visual_8509 import theme,hero
from judge_release import require_judge_access,seed_release_users,judge_mode

st.set_page_config(page_title='WEB&SHOP · Visual inventory',page_icon='W',layout='wide')
theme()
require_judge_access()
db.init_db();init_registration()
business=seed_release_users();db.default_location(business['id']);db.migrate_catalog_json(business['id'])
if 'user_id' not in st.session_state:
    hero('WEB&SHOP', 'From a product photo to a reviewed shelf count. OpenCV 5 · AWS · NVIDIA DINO · SigLIP2')
    role=st.radio('Role view',['Owner / employee sign in','Buyer'],horizontal=True)
    if role=='Buyer':
        from buyer_8509 import buyer_view
        buyer_view();st.stop()
    with st.container(border=True):
        st.subheader('Welcome to your shop')
        with st.form('login_8509'):
            name=st.text_input('Username',key='login_username')
            password=st.text_input('Password',type='password',key='login_password')
            if st.form_submit_button('Sign in',type='primary'):
                user=auth.authenticate(name,password)
                if user:st.session_state['user_id']=user['id'];st.rerun()
                else:st.error('Incorrect username or password.')
    st.stop()
service=InventoryService(st.session_state['user_id'])
try:user=service.user()
except PermissionError:
    st.session_state.clear();st.rerun()
with st.sidebar:
    st.title('WEB&SHOP')
    st.write(db.get_business(user['business_id'])['name'])
    st.caption(f"{user['username']} · {user['role']}")
    choices=['Owner','Employee Scan','Buyer'] if user['role']=='owner' else ['Employee Scan','Buyer']
    page=st.radio('Workspace',choices)
    st.caption('JUDGE SANDBOX · fictional data' if judge_mode() else 'LOCAL DEMO · fictional data')
    st.caption('Review first. Approve stock changes separately.')
    if st.button('Sign out'):
        st.session_state.clear();st.rerun()
if st.session_state.get('flash'):st.success(st.session_state.pop('flash'))
if page=='Owner':
    from owner_8509 import dashboard
    dashboard(service)
elif page=='Employee Scan':
    hero('See it. Review it. Count it.', 'AWS/OpenCV capture guidance → live Retail DINO boxes → current-shop SigLIP2 → your approval')
    from scanner_8509 import fresh_scan
    fresh_scan(service)
else:
    from buyer_8509 import buyer_view
    buyer_view()
