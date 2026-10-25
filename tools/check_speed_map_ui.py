"""Actual browser map/calculator check, without inference or inventory writes."""
import json
import sqlite3
import time
from pathlib import Path
from capture_8509_visuals import Browser,ROOT


def ledger():
    with sqlite3.connect(ROOT/'data/webshop_ux_demo/inventorylens.sqlite3') as c:
        return {t:c.execute('SELECT * FROM '+t+' ORDER BY id').fetchall() for t in ('scan_sessions','scan_items','inventory_counts')}


before=ledger()
b=Browser()
b.call('Page.navigate',dict(url='http://localhost:8509'))
time.sleep(1)
b.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
b.wait('Role view');b.radio('Buyer');b.wait('Search the catalogue')
b.fill('Search the catalogue','Freshona Lentils')
b.wait('Shopping quantity');b.fill('Shopping quantity','2')
b.radio('Shopping list & map');b.wait('Ordered shelf list only');b.idle()
assert 'Next stop: A1' in b.js('document.body.innerText')
assert '2.98' in b.js('document.body.innerText')
b.screenshot('webshop_8509_buyer_map.png')
b.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Shopping stops'))?.scrollIntoView({block:'start'})")
b.screenshot('webshop_8509_buyer_map_mobile.png')
assert b.js('document.documentElement.scrollWidth')==b.js('window.innerWidth')
assert ledger()==before
report=dict(actual_browser=True,buyer_list_quantity=2,subtotal_eur=2.98,highlighted_shelf='A1',fictional_workbook_shelf_count=8,
            shortest_route_claimed=False,stock_unchanged=True,mobile_width=390,mobile_overflow=False)
(ROOT/'data/webshop_ux_demo/map_browser_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));b.ws.close()
