"""Owner preview and explicit SKU reconciliation around the existing importer."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import db
from shop_catalog import read_catalog_bundle, validate_catalog, validate_shelves


def catalog_import(service):
    user = service.user(owner=True)
    st.caption('Catalogue values are shared by this shop. Fictional workbook stock is a demo opening balance.')
    with st.expander('Import catalogue workbook / CSV'):
        upload = st.file_uploader('Products / Products_Import XLSX or catalogue CSV', type=['xlsx', 'csv'], key='catalog_import')
        sample = Path(__file__).resolve().parent / 'demo_data/WEB_SHOP_mock_catalog_prices_shelves_map.xlsx'
        use_sample = st.checkbox('Preview supplied fictional workbook', disabled=not sample.exists())
        if not upload and not use_sample:
            return
        try:
            data, name = (upload.getvalue(), getattr(upload, 'name', 'catalogue.xlsx')) if upload else (sample.read_bytes(), sample.name)
            digest = hashlib.sha256(data).hexdigest()[:16]
            bundle = read_catalog_bundle(data, name)
            if bundle['fictional']:
                st.warning('Fictional demo workbook. Identity mappings remain pending; spreadsheet IDs do not register recognition references.')
                if not db.get_business(user['business_id']).get('is_demo'):
                    st.error('Use the isolated UX demo shop for this workbook. Launch tools/start_webshop_ux_demo.ps1 on port 8509.')
            existing = {p['sku']: p for p in service.products()}
            rows = []
            fields = ['sku', 'name', 'price', 'initial_quantity', 'shelf_id', 'unit', 'brand_variant_size', 'registration_status', 'identity_status']
            for row in bundle['products']:
                rows.append({**{k: row.get(k, '') for k in fields}, 'include': True,
                             'resolution': 'Existing shop SKU' if row.get('sku', '').upper() in existing else 'Pending new ID'})
            st.caption('Edit SKU to map to an existing product, or retain a pending new ID. Remove duplicates or uncheck Include. Existing reference images are preserved.')
            frame = st.data_editor(pd.DataFrame(rows), hide_index=True, num_rows='dynamic', width='stretch',
                key=f'import_products_{digest}', column_config={
                    'resolution': st.column_config.SelectboxColumn('SKU resolution', options=['Existing shop SKU', 'Pending new ID']),
                    'include': st.column_config.CheckboxColumn('Include')})
            selected = frame.fillna('').to_dict('records')
            records, errors = [], []
            for number, row in enumerate(selected, 1):
                if row.pop('include', False) is not True:
                    continue
                resolution = row.pop('resolution', '')
                sku = str(row.get('sku', '')).strip().upper()
                if resolution not in {'Existing shop SKU', 'Pending new ID'}:
                    errors.append(f'Row {number}: choose an explicit SKU resolution.')
                if resolution == 'Existing shop SKU' and sku not in existing:
                    errors.append(f'Row {number}: {sku} is not an existing current-shop SKU.')
                if resolution == 'Pending new ID' and sku in existing:
                    errors.append(f'Row {number}: {sku} already exists; choose Existing shop SKU.')
                try:
                    validate_catalog([row])
                except (ValueError, TypeError) as error:
                    errors.append(f'Row {number}: {error}')
                records.append(row)
            shelf_rows = bundle['shelves']
            if shelf_rows:
                st.subheader('Shelf locations from workbook')
                shelf_rows = st.data_editor(pd.DataFrame(shelf_rows), hide_index=True, num_rows='dynamic',
                    width='stretch', key=f'import_shelves_{digest}').fillna('').to_dict('records')
                validate_shelves(shelf_rows)
                known_shelves = {str(r['shelf_id']).strip() for r in shelf_rows}
                for row in records:
                    if row.get('shelf_id') and row['shelf_id'] not in known_shelves:
                        errors.append(f"{row['sku']}: unknown shelf {row['shelf_id']}.")
            try:
                preview = validate_catalog(records)
            except (ValueError, TypeError) as error:
                preview = []
                errors.append(str(error))
            for error in dict.fromkeys(errors):
                st.error(error)
            if errors:
                return
            st.subheader('Validated import preview')
            st.dataframe(preview, hide_index=True, width='stretch')
            opening = st.checkbox('Initialize fictional opening stock only where no balance exists')
            approval_key = hashlib.sha256(json.dumps([records, shelf_rows, opening], sort_keys=True).encode()).hexdigest()[:16]
            accepted = st.checkbox('I reviewed SKU mappings, prices, stock and shelves; new IDs remain unregistered', key=f'import_confirm_{approval_key}')
            allowed_demo = not bundle['fictional'] or bool(db.get_business(user['business_id']).get('is_demo'))
            if st.button('Import catalogue', disabled=not accepted or not allowed_demo):
                count = service.import_catalog(records, confirmed=accepted, opening=opening,
                                               shelves=shelf_rows, fictional=bundle['fictional'])
                st.session_state['flash'] = f'Imported {count} catalogue entries. Existing reference images were preserved.'
                st.rerun()
        except Exception as error:
            st.error(f'Import validation: {error}')
