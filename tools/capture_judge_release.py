"""Actual public-safe UI screenshots; no private photos and no model calls."""
import json
from pathlib import Path
import sqlite3
import time
from browser_evidence import Browser

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'docs/public';folder.mkdir(parents=True,exist_ok=True)
    access=json.loads((ROOT/'data/judge_release/access.json').read_text())
    def ledger():
        with sqlite3.connect(ROOT/'data/webshop_judge_demo/inventorylens.sqlite3') as c:
            return {t:c.execute('SELECT * FROM '+t+' ORDER BY id').fetchall() for t in ('scan_sessions','scan_items','inventory_counts')}
    before=ledger();b=Browser();b.viewport()
    b.call('Page.navigate',dict(url='http://localhost:8511'));time.sleep(1)
    b.wait('Judge access code');b.fill('Judge access code','incorrect-code');b.click('Open judge demo');b.wait('Access code not accepted.')
    b.fill('Judge access code',access['access_code']);b.click('Open judge demo');b.wait('Welcome to your shop')
    b.fill('Username','owner');b.fill('Password','owner123');b.click('Sign in');b.wait('Incorrect username or password.')
    b.fill('Username',access['owner_username']);b.fill('Password',access['owner_password']);b.click('Sign in')
    b.wait('Find a product');b.wait('Test Recognition');b.idle();b.screenshot(folder/'owner.png')
    b.click('+ Add Product');b.wait('Product name');b.idle();b.screenshot(folder/'registration.png')
    b.viewport(390,844);b.screenshot(folder/'registration-mobile.png');b.viewport()
    b.click('Sign out');b.wait('Judge access code');b.fill('Judge access code',access['access_code']);b.click('Open judge demo')
    b.wait('Username');b.fill('Username',access['employee_username']);b.fill('Password',access['employee_password']);b.click('Sign in')
    b.wait('Take a new photograph');b.idle()
    assert 'Archived photo' not in b.js('document.body.innerText')
    b.screenshot(folder/'employee.png')
    b.radio('Buyer');b.wait('Search the catalogue');b.fill('Search the catalogue','Energia')
    b.wait('Shopping quantity')
    deadline=time.monotonic()+30
    while b.js("document.querySelectorAll('input[aria-label=\"Shopping quantity\"]').length")!=1:
        if time.monotonic()>deadline:raise RuntimeError('Catalogue search did not settle to one product.')
        time.sleep(.3)
    b.fill('Shopping quantity','2');b.wait('\u20ac3.18');b.screenshot(folder/'buyer.png')
    b.radio('Shopping list & map');b.wait('Next stop: C1');b.idle()
    b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Shopping stops')).scrollIntoView({block:'start'})")
    b.screenshot(folder/'buyer-map.png')
    b.viewport(390,844);b.screenshot(folder/'buyer-mobile.png')
    assert b.js('document.documentElement.scrollWidth')==b.js('window.innerWidth')
    assert ledger()==before
    result=dict(actual_browser=True,private_photos_copied=False,default_credentials_rejected=True,wrong_gate_code_rejected=True,private_replay_disabled=True,
                buyer_stock_unchanged=True,mobile_overflow=False,public_https_verified=False,reason='Local checks only. External TLS/email-gate and authenticated acceptance are recorded separately.')
    (ROOT/'data/judge_release/browser_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    b.ws.close()


if __name__=='__main__':main()
