"""Buyer browse/list/cart preview using the authoritative shop catalogue."""
import json

import streamlit as st

import db
from shop_catalog import BuyerCatalog, shopping_list
from scan_review import price_estimate


def show_price_estimate(products, quantities, key):
    estimate = price_estimate(products, quantities)
    st.dataframe(estimate['lines'], hide_index=True, width='stretch')
    st.metric('Estimated priced subtotal', f"€{estimate['total']:.2f}")
    if not estimate['complete']:
        st.warning('Incomplete total — unknown products or missing prices excluded: ' + ', '.join(estimate['excluded']))
    if st.checkbox('Set a budget', key=f'{key}_budget_enabled'):
        budget = st.number_input('Budget (€)', min_value=0.0, value=20.0, step=1.0, key=f'{key}_budget')
        remaining = round(budget - estimate['total'], 2)
        st.metric('Remaining' if remaining >= 0 else 'Over budget', f'€{abs(remaining):.2f}')
        if not estimate['complete']:
            st.caption('Budget comparison covers priced items only.')
    return estimate


def buyer_view():
    st.header('WEB&SHOP · Buyer')
    with db.connect() as connection:
        shops = db.rows(connection.execute('SELECT b.id,b.name FROM businesses b WHERE EXISTS '
                                          '(SELECT 1 FROM products p WHERE p.business_id=b.id) ORDER BY b.name'))
    if not shops:
        st.info('An owner must create a shop first.')
        return
    shop = st.selectbox('Shop', [s['id'] for s in shops], format_func=lambda key: next(s['name'] for s in shops if s['id'] == key))
    service = BuyerCatalog(shop)
    products = service.products()
    st.caption('Catalogue prices and shelf locations are owner-entered. Demo shop values are fictional. Blank values are pending.')
    search = st.text_input('Search products').casefold()
    visible = [p for p in products if search in (p['sku'] + ' ' + p['name']).casefold()]
    st.dataframe([{k: p.get(k) for k in ('sku', 'name', 'price', 'unit', 'shelf_id', 'registration_status')} for p in visible], hide_index=True)
    st.subheader('Shelf locations')
    st.dataframe([{'Shelf': l['name'], 'Products': ', '.join(p['name'] for p in products if p.get('shelf_id') == l['name'])}
                  for l in service.locations()], hide_index=True)
    st.subheader('Shopping list')
    lists = st.session_state.setdefault('buyer_lists', {})
    quantities = lists.setdefault(shop, {})
    for p in visible:
        quantities[p['sku']] = st.number_input(f"{p['sku']} · {p['name']}", min_value=0, max_value=10000,
                                               value=quantities.get(p['sku'], 0), step=1, key=f"list_{shop}_{p['sku']}")
    items = shopping_list(products, quantities)
    show_price_estimate(products, quantities, f'list_{shop}')
    st.download_button('Download shopping list', json.dumps(items, indent=2), 'shopping_list.json')
    from shop_map import buyer_map
    buyer_map(service.business_id, [item['shelf_id'] for item in items if item['shelf_id']])
    if st.checkbox('Camera-assisted cart preview'):
        from scanner_ui import fresh_scan
        fresh_scan(service, buyer=True)
