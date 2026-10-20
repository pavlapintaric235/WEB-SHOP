"""Owner dashboard using the same authorized service as the scanner."""

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import Image, ImageOps

import db
from images import product_references


def local_times(items):
    for item in items:
        for field in ("created_at", "updated_at"):
            if item.get(field):
                item[field] = datetime.fromisoformat(item[field]).astimezone(
                    ZoneInfo("Europe/Zagreb")
                ).strftime("%Y-%m-%d %H:%M:%S %Z")
    return items


def dashboard(service, read_image):
    service.user(owner=True)
    inventory_tab, products_tab, team_tab = st.tabs(["Inventory", "Products", "Team"])
    with inventory_tab:
        with st.form("new_location"):
            name = st.text_input("New location", placeholder="Shelf A")
            if st.form_submit_button("Add location"):
                try:
                    service.add_location(name)
                    st.rerun()
                except (ValueError, PermissionError) as error:
                    st.error(str(error))
        locations = service.locations()
        location_id = st.selectbox(
            "Inventory location", [item["id"] for item in locations],
            format_func=lambda value: next(item["name"] for item in locations if item["id"] == value),
        )
        if st.button("Refresh inventory", icon=":material/refresh:"):
            st.rerun()
        if location_id is not None:
            st.dataframe(local_times(service.inventory(location_id)), hide_index=True, use_container_width=True)
            st.download_button(
                "Download location CSV", service.export_csv(location_id),
                file_name=f"inventory-location-{location_id}.csv", mime="text/csv",
                icon=":material/download:",
            )
            with st.expander("Import inventory CSV"):
                upload = st.file_uploader("CSV with sku and quantity", type=["csv"], key="owner_csv")
                if upload:
                    try:
                        text = upload.getvalue().decode("utf-8-sig")
                        _, parsed = db.validate_csv_text(text)
                        st.dataframe(parsed, hide_index=True, use_container_width=True)
                        confirmed = st.checkbox(
                            "Replace matching SKU quantities at this location with the imported values",
                            key=f"confirm_csv_{location_id}",
                        )
                        if st.button("Import CSV", disabled=not confirmed, icon=":material/upload:"):
                            count = service.import_csv(location_id, upload.name, text, confirmed)
                            st.session_state["flash"] = f"Imported {count} registered SKU quantities."
                            st.rerun()
                    except (ValueError, UnicodeDecodeError, csv.Error, PermissionError) as error:
                        st.error(str(error))
        st.subheader("Approved count history")
        history = local_times(service.history())
        if history:
            st.dataframe(history, hide_index=True, use_container_width=True)
        else:
            st.info("No approved counts yet.")

    with products_tab:
        with st.form("register", clear_on_submit=True):
            sku = st.text_input("SKU", placeholder="COLA-330")
            name = st.text_input("Product name")
            uploads = st.file_uploader("Reference photographs", type=["png", "jpg", "jpeg"],
                                       accept_multiple_files=True, key="new_product_references",
                                       help=f"Maximum {db.MAX_PRODUCT_REFERENCES} photographs per product.")
            if st.form_submit_button("Save reference", type="primary"):
                try:
                    service.register_product(sku, name, [read_image(upload) for upload in uploads])
                    st.session_state["flash"] = f"Registered {sku.strip().upper()}."
                    st.rerun()
                except (ValueError, OSError, PermissionError) as error:
                    st.error(str(error))
        products = service.products()
        if products:
            st.dataframe(
                [{"SKU": item["sku"], "Product": item["name"],
                  "References": len(product_references(item))} for item in products],
                hide_index=True, use_container_width=True,
            )
            by_id = {item["id"]: item for item in products}
            selected_id = st.selectbox("Product", list(by_id), key="selected_product",
                                       format_func=lambda value: f'{by_id[value]["sku"]} - {by_id[value]["name"]}')
            selected = by_id[selected_id]
            references = product_references(selected)
            st.subheader("Reference photographs")
            for start in range(0, len(references), 3):
                columns = st.columns(3)
                for index, reference in enumerate(references[start:start + 3], start=start):
                    with columns[index % 3]:
                        try:
                            with Image.open(Path(reference["image_path"])) as source:
                                thumbnail = ImageOps.pad(ImageOps.exif_transpose(source).convert("RGB"),
                                                         (240, 240), color="white")
                            st.image(thumbnail, width="stretch", caption=f"View {index + 1}")
                        except OSError:
                            st.error("Reference photograph is missing or unreadable.")
                        if st.button("Remove reference", icon=":material/delete:",
                                     key=f'remove_reference_{reference["id"]}', disabled=len(references) == 1):
                            try:
                                service.remove_reference(selected["sku"], reference["id"])
                                st.session_state["flash"] = "Reference removed."
                                st.rerun()
                            except (ValueError, PermissionError) as error:
                                st.error(str(error))
            with st.form(f'add_references_{selected["id"]}', clear_on_submit=True):
                additional = st.file_uploader("Additional reference photographs", type=["png", "jpg", "jpeg"],
                                               accept_multiple_files=True,
                                               key=f'additional_references_{selected["id"]}',
                                               help=f'{len(references)} of {db.MAX_PRODUCT_REFERENCES} photographs saved.',
                                               disabled=len(references) >= db.MAX_PRODUCT_REFERENCES)
                if st.form_submit_button("Add references", icon=":material/add_photo_alternate:",
                                         disabled=len(references) >= db.MAX_PRODUCT_REFERENCES):
                    try:
                        count = service.add_references(selected["sku"], [read_image(upload) for upload in additional])
                        st.session_state["flash"] = f"Added {count} reference photograph(s)."
                        st.rerun()
                    except (ValueError, OSError, PermissionError) as error:
                        st.error(str(error))
            with st.form(f'edit_{selected["id"]}'):
                updated_name = st.text_input("Product name", value=selected["name"])
                if st.form_submit_button("Update product"):
                    try:
                        service.update_product(selected["sku"], updated_name)
                        st.session_state["flash"] = "Product updated."
                        st.rerun()
                    except (ValueError, PermissionError) as error:
                        st.error(str(error))

    with team_tab:
        st.dataframe(local_times(service.employees()), hide_index=True, use_container_width=True)
        with st.form("add_employee", clear_on_submit=True):
            username = st.text_input("Employee username")
            password = st.text_input("Initial password", type="password")
            if st.form_submit_button("Add employee"):
                try:
                    service.add_employee(username, password)
                    st.session_state["flash"] = "Employee account created."
                    st.rerun()
                except (ValueError, PermissionError) as error:
                    st.error(str(error))
