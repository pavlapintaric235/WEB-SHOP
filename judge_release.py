"""Opt-in protections for an isolated, access-controlled judge sandbox."""
import hmac
import os
from pathlib import Path
import re
import time


def judge_mode():
    return os.environ.get('WEBSHOP_JUDGE_MODE')=='1'


def sandbox_directory():
    return Path(__file__).resolve().parent/'data/webshop_judge_demo'


def validate_sandbox():
    import auth,db
    expected=sandbox_directory()
    if db.DATA_DIR.resolve()!=expected.resolve():
        raise RuntimeError('Judge mode requires its own prepared sandbox database.')
    owner=auth.get_user('judge-owner')
    if not owner:raise RuntimeError('Run tools/prepare_judge_demo.py before starting the judge sandbox.')
    with db.connect() as c:
        users=db.rows(c.execute('SELECT * FROM users'))
        paths=[r['image_path'] for r in c.execute('SELECT image_path FROM products WHERE image_path <> \'\'')]
        paths.extend(r['image_path'] for r in c.execute('SELECT image_path FROM product_references'))
    if any(auth.verify_password(password,u['password_hash']) for u in users for password in ('owner123','employee123')):
        raise RuntimeError('Default demo passwords are forbidden in judge mode.')
    if any(not Path(p).resolve().is_relative_to(expected.resolve()) for p in paths):
        raise RuntimeError('Judge references must be uploaded into the sandbox; private local references cannot be reused.')
    return auth.get_business(owner['business_id'])


def seed_release_users():
    import auth
    return validate_sandbox() if judge_mode() else auth.seed_demo_users()


def require_judge_access():
    if not judge_mode():return
    import streamlit as st
    code=os.environ.get('WEBSHOP_JUDGE_ACCESS_CODE','')
    if len(code)<24:raise RuntimeError('A strong separate judge access code is required.')
    if st.session_state.get('_judge_access_until',0)>time.time():return
    if st.session_state.pop('_judge_access_until',None):
        st.session_state.clear()
    st.title('WEB&SHOP')
    st.caption('Private judge sandbox · fictional catalogue · individual role sign-in follows')
    with st.form('judge_access_gate'):
        supplied=st.text_input('Judge access code',type='password')
        if st.form_submit_button('Open judge demo',type='primary'):
            if hmac.compare_digest(supplied,code):
                st.session_state['_judge_access_until']=time.time()+1800;st.rerun()
            else:st.error('Access code not accepted.')
    st.caption('An email-restricted HTTPS gateway must protect the whole hostname, including images and WebSockets. Access expires after 30 minutes.')
    st.stop()


def user_error(error):
    text=str(error)
    if judge_mode() and (not isinstance(error,(ValueError,PermissionError)) or re.search(r'https?://|[A-Za-z]:[\\/]|/workspace|/Users/|token|secret',text,re.I)):
        return 'The action could not finish. Retry or contact the demo operator; no automatic fallback was applied.'
    return text


def shareable_evidence(value):
    if not judge_mode():return value
    private={'source_bucket','source_key','replay_archive','image_path','original_path','normalized_path','embedding_path','token','url','authorization'}
    if isinstance(value,dict):return {k:shareable_evidence(v) for k,v in value.items() if k.lower() not in private}
    if isinstance(value,list):return [shareable_evidence(v) for v in value]
    if isinstance(value,str) and re.search(r'https?://|[A-Za-z]:[\\/]|/workspace',value):return '[operator detail omitted]'
    return value
