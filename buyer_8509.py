"""Buyer product cards using the same catalogue, calculator and manual map."""
import json
import streamlit as st
import db
from shop_catalog import BuyerCatalog, shopping_list
from images import product_references
from buyer_ui import show_price_estimate
from visual_8509 import hero,thumbnail,photo_placeholder


def buyer_view():
    hero('A clearer trip around your shop.', 'Plan your list, see an estimated budget, and find the owner-assigned shelves.')
    with db.connect() as c:
        shops=db.rows(c.execute('SELECT id,name FROM businesses WHERE EXISTS(SELECT 1 FROM products WHERE business_id=businesses.id)'))
    if not shops:st.info('The owner has not registered a catalogue yet.');return
    shop=st.selectbox('Shop',[s['id'] for s in shops],format_func=lambda x:next(s['name'] for s in shops if s['id']==x))
    service=BuyerCatalog(shop);products=service.products()
    st.caption('Prices and locations come from this shop. Demo prices are fictional, not a checkout quote.')
    section=st.radio('Buyer section',['Products','Shopping list & map','Camera cart'],horizontal=True,key='buyer_section_8509')
    search=st.text_input('Search the catalogue').casefold()
    lists=st.session_state.setdefault('buyer_lists_8509',{});quantities=lists.setdefault(shop,{})
    columns=st.columns(3)
    for index,p in enumerate([p for p in products if section=='Products' and search in (p['name']+' '+p['sku']).casefold()]):
        with columns[index%3],st.container(border=True,key=f'product_card_buyer_{shop}_{p["id"]}'):
            refs=product_references(p)
            if refs:st.image(thumbnail(refs[0]['image_path']),width='stretch')
            else:photo_placeholder(p['name'])
            st.markdown(f"#### {p['name']}")
            st.markdown(f"**€{p['price']:.2f}**" if p.get('price') is not None else '**Price unavailable**')
            st.caption(f"{p['sku']} · Shelf {p.get('shelf_id') or 'not assigned'}")
            quantities[p['sku']]=st.number_input('Shopping quantity',0,10000,int(quantities.get(p['sku'],0)),key=f'buyer_card_{shop}_{p["sku"]}')
    items=shopping_list(products,quantities)
    from shop_map import buyer_map
    if section=='Shopping list & map':
        map_column,list_column=st.columns([1.3,1],gap='large')
        with map_column:buyer_map(shop,[i['shelf_id'] for i in items if i['shelf_id']],show_empty=True)
    else:list_column=st.container()
    with list_column:
        shopping_summary(products,quantities,items,shop,section)
    if section!='Shopping list & map':
        st.caption('Open Shopping list & map above to see the 2D shelf layout and your shopping stops.')
    if section=='Camera cart' or st.checkbox('Camera-assisted cart preview'):
        from scanner_8509 import fresh_scan
        fresh_scan(service,buyer=True)


def shopping_summary(products,quantities,items,shop,section):
    st.subheader('Your shopping list')
    if section=='Shopping list & map':
        st.caption('Choose quantities in Products, then return here. Yellow shelves contain items on your list.')
        if items:st.dataframe(items,hide_index=True,width='stretch')
        missing=[i['name'] for i in items if not i.get('shelf_id')]
        if missing:st.warning('No shelf assigned for: '+', '.join(missing)+'. Ask the owner to assign a shelf; these items cannot be placed on the map.')
    show_price_estimate(products,quantities,f'cards_{shop}')
    st.download_button('Save shopping list',json.dumps(items,indent=2),'shopping_list.json')
