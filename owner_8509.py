"""Owner cards and guided registration, reusing the existing authorized services."""
import streamlit as st
from judge_release import user_error

from images import product_references
from visual_8509 import hero,thumbnail,photo_placeholder
from registration_8509 import export_catalog, update_details


def dashboard(service, read_image=None):
    service.user(owner=True)
    hero('Your products. Ready for the shelf.', 'Register a package from photos, keep the catalogue current, and review every stock change.')
    if st.button('+ Add Product',type='primary',key='add_product_primary'):
        st.session_state['owner_section_8509']='Add Product'
    section=st.radio('Owner workspace',['Products','Add Product','Inventory','Shelves','Employees','Reports'],horizontal=True,key='owner_section_8509')
    products=service.products()
    if section=='Add Product':
        from registration_wizard_8509 import wizard
        wizard(service);return
    if section=='Products':
        registered=sum(bool(product_references(p)) for p in products)
        stats=st.columns(3)
        stats[0].metric('Products',len(products))
        stats[1].metric('With reference photos',registered)
        stats[2].metric('Awaiting photos',len(products)-registered)
        from catalog_import_ui import catalog_import
        catalog_import(service)
        search=st.text_input('Find a product',placeholder='Name, brand or SKU').casefold()
        visible=[p for p in products if search in (p['name']+' '+p['sku']+' '+(p.get('brand') or '')).casefold()]
        counts=[r for l in service.locations() for r in service.inventory(l['id'])]
        columns=st.columns(3)
        for i,p in enumerate(visible):
            with columns[i%3],st.container(border=True,key=f'product_card_{p["id"]}'):
                references=product_references(p)
                if references:st.image(thumbnail(references[0]['image_path']),width='stretch')
                else:photo_placeholder(p['name'])
                st.markdown(f"#### {p['name']}")
                st.caption(p['sku'])
                st.markdown(f"**€{p['price']:.2f}**" if p.get('price') is not None else '**Price pending**')
                stock=sum(r['quantity'] or 0 for r in counts if r['sku']==p['sku'])
                st.caption(f"Shelf {p.get('shelf_id') or 'pending'} · {stock} units · {len(references)} photos")
                st.caption('References available · review suggestions before counting' if references else 'Upload 3–5 approved photos to enable recognition')
                actions=st.columns(2)
                for col,label in zip(actions,['View / Edit','Add Photos']):
                    if col.button(label,key=f"{label}_{p['id']}"):
                        st.session_state['product_selected_8509']=p['id'];st.session_state['product_action_8509']=label
                if st.button('Test Recognition',key=f"test_{p['id']}"):
                    st.session_state['product_selected_8509']=p['id'];st.session_state['product_action_8509']='Test Recognition'
        selected=next((p for p in products if p['id']==st.session_state.get('product_selected_8509')),None)
        if selected:
            st.divider();st.subheader(selected['name'])
            action=st.session_state.get('product_action_8509')
            if action=='Add Photos':
                from registration_wizard_8509 import wizard
                wizard(service,selected)
            elif action=='Test Recognition':
                from owner_test_8509 import test_recognition
                test_recognition(service,selected)
            else:
                with st.form('edit_product_8509'):
                    price=st.number_input('Price in EUR',min_value=0.0,value=float(selected.get('price') or 0),step=.1)
                    shelves=[l['name'] for l in service.locations()]
                    shelf=st.selectbox('Shelf / location',shelves,index=shelves.index(selected['shelf_id']) if selected.get('shelf_id') in shelves else 0)
                    st.caption(selected.get('brand_variant_size') or 'Package details pending')
                    if st.form_submit_button('Save product details',type='primary'):
                        try:update_details(service,selected['id'],price,shelf);st.rerun()
                        except (ValueError,PermissionError) as error:st.error(user_error(error))
        return
    if section=='Inventory':
        locations=service.locations()
        location=st.selectbox('Shelf / count location',[l['id'] for l in locations],format_func=lambda x:next(l['name'] for l in locations if l['id']==x))
        st.dataframe(service.inventory(location),hide_index=True,width='stretch')
        st.download_button('Export shelf counts',service.export_csv(location),'shelf_counts.csv')
    elif section=='Shelves':
        with st.form('add_shelf_8509'):
            name=st.text_input('New shelf ID')
            if st.form_submit_button('Add shelf'):
                try:service.add_location(name);st.rerun()
                except (ValueError,PermissionError) as error:st.error(user_error(error))
        from shop_map import owner_map
        owner_map(service)
    elif section=='Employees':
        st.dataframe(service.employees(),hide_index=True,width='stretch')
        with st.form('employee_8509'):
            name=st.text_input('Employee username');password=st.text_input('Initial password',type='password')
            if st.form_submit_button('Add employee'):
                try:service.add_employee(name,password);st.rerun()
                except (ValueError,PermissionError) as error:st.error(user_error(error))
        st.subheader('Product registration requests')
        st.dataframe(service.registration_requests(),hide_index=True,width='stretch')
    elif section=='Reports':
        st.download_button('Export current product catalogue CSV',export_catalog(service),'webshop_current_catalogue.csv',mime='text/csv')
        st.caption('Export is generated from the authoritative database. Your imported workbook is preserved.')
        st.dataframe(service.history(),hide_index=True,width='stretch')
