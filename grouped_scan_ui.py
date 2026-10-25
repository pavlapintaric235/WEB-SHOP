"""Streamlit review controls around durable, stable-ID scan drafts."""
import streamlit as st

from live_scan import render_items, valid_box
from scan_review import add_manual, edit_item, grouped, set_quantity, split_item


def overlay_rows(draft):
    return [dict(box=[r[k] for k in ('x1','y1','x2','y2')], **{k: r[k] for k in ('sku','item_id','confirmed','origin')})
            for r in draft['rows'] if not r['exclude'] and r['box_source'] != 'unlocalized']


def review_panel(image, products, draft, actor, prefix, locked=False):
    names = {p['sku']: p['name'] for p in products}
    choices = [''] + list(names)
    label = lambda sku: f'{names[sku]} · {sku}' if sku else 'Unknown — owner registration needed'
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        st.image(render_items(image, overlay_rows(draft)), caption='Stable box IDs · green reviewed · amber pending · blue manual', width='stretch')
        unlocalized = [r for r in draft['rows'] if not r['exclude'] and r['box_source'] == 'unlocalized']
        if unlocalized:
            st.caption('Manual quantity additions have no detected box: ' + ', '.join(r['item_id'] for r in unlocalized))
    with right:
        st.subheader('Products in this photo')
        if not locked:
            with st.expander('Add missed product', expanded=not bool(draft['rows'])):
                sku = st.selectbox('Search current catalogue', list(names), format_func=label, key=f'{prefix}_miss_sku')
                quantity = st.number_input('Missed units', 1, 10000, 1, key=f'{prefix}_miss_qty')
                if st.button('Add missed product', key=f'{prefix}_miss_add'):
                    add_manual(draft, sku, int(quantity), image.size, actor)
                    st.rerun()
        for sku, group in grouped(draft).items():
            group_key = f'{prefix}_{sku or "unknown"}_{draft["revision"]}'
            st.markdown(f'**{label(sku)}**')
            st.caption(f"{group['quantity']} units · {'Reviewed' if group['reviewed'] else 'Needs review'} · IDs " + ', '.join(r['item_id'] for r in group['rows']))
            if sku and not locked:
                minus, quantity_col, plus, apply = st.columns([1, 2, 1, 2])
                with minus:
                    if st.button('−', key=f'{group_key}_minus'):
                        set_quantity(draft, sku, max(0, group['quantity']-1), image.size, actor)
                        st.rerun()
                with quantity_col:
                    quantity = st.number_input('Units', 0, 10000, int(group['quantity']), key=f'{group_key}_units', label_visibility='collapsed')
                with plus:
                    if st.button('+', key=f'{group_key}_plus', disabled=group['quantity'] >= 10000):
                        set_quantity(draft, sku, group['quantity']+1, image.size, actor)
                        st.rerun()
                with apply:
                    if st.button('Set quantity', key=f'{group_key}_set'):
                        set_quantity(draft, sku, int(quantity), image.size, actor)
                        st.rerun()
                if st.button('Mark group reviewed', key=f'{group_key}_review', disabled=group['reviewed']):
                    for row in group['rows']:
                        edit_item(draft, row['item_id'], {'confirmed': True}, actor)
                    st.rerun()
            with st.expander(f"Inspect / correct {label(sku)} boxes"):
                for row in group['rows']:
                    key = f'{prefix}_item_{row["item_id"]}_{row.get("revision", 0)}'
                    st.markdown(f"**Box {row['item_id']}** · {row['origin']}" + (f" · split from {row['parent_id']}" if row['parent_id'] else ''))
                    if row.get('suggestions'):
                        st.caption('Model suggestions: ' + ', '.join(f"{r['sku']} ({r['similarity']:.2f})" for r in row['suggestions'][:3]))
                    if locked:
                        st.json(row)
                        continue
                    with st.form(key):
                        selected = st.selectbox('Product for this box', choices, index=choices.index(row['sku']) if row['sku'] in choices else 0, format_func=label)
                        units = st.number_input('Units in this box', 1, 10000, int(row['quantity']))
                        coords = st.columns(4)
                        values = [coords[i].number_input(k, value=float(row[k]), min_value=0.0) for i, k in enumerate(('x1','y1','x2','y2'))]
                        confirmed = st.checkbox('I reviewed this instance', value=row['confirmed'])
                        if st.form_submit_button('Save box correction'):
                            try:
                                valid_box(values, image.size)
                                if confirmed and not selected:
                                    raise ValueError('Unknown products need owner registration; choose a known SKU only after checking its identity.')
                                changes = dict(sku=selected, quantity=int(units), confirmed=confirmed,
                                               **dict(zip(('x1','y1','x2','y2'), values)))
                                edit_item(draft, row['item_id'], changes, actor)
                                st.rerun()
                            except ValueError as error:
                                st.error(str(error))
                    if st.button('Remove false detection', key=f'{key}_remove'):
                        edit_item(draft, row['item_id'], {'exclude': True}, actor)
                        st.rerun()
                    if sku:
                        parts = st.number_input('Split into instances', 2, 8, 2, key=f'{key}_parts')
                        st.caption('Split proposes equal vertical boxes. Inspect and correct each child before marking it reviewed.')
                        if st.button('Split merged detection', key=f'{key}_split'):
                            width = (row['x2']-row['x1']) / parts
                            children = [dict(sku=sku, box=[row['x1']+i*width, row['y1'], row['x1']+(i+1)*width, row['y2']]) for i in range(parts)]
                            split_item(draft, row['item_id'], children, image.size, actor)
                            st.rerun()
        removed = [r for r in draft['rows'] if r['exclude']]
        if removed:
            with st.expander(f'Removed / split parent detections ({len(removed)})'):
                for row in removed:
                    st.caption(f"{row['item_id']} · {label(row['sku'])}")
                    # Split parents must stay excluded while their children exist.
                    children = any(r['parent_id'] == row['item_id'] and not r['exclude'] for r in draft['rows'])
                    if not locked and not children and st.button('Restore detection', key=f'{prefix}_restore_{row["item_id"]}'):
                        edit_item(draft, row['item_id'], {'exclude': False, 'confirmed': False}, actor)
                        st.rerun()
