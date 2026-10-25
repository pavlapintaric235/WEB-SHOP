"""Atomic stock actions for a reviewed photo spanning assigned catalogue shelves."""
import db


def _preview(connection, shop, counts, operation):
    if operation not in ('add','subtract'):
        raise ValueError('Assigned-shelf actions support ADD or SUBTRACT.')
    if not counts:
        raise ValueError('Choose at least one reviewed product.')
    products={r['sku']:dict(r) for r in connection.execute('''
        SELECT p.sku,p.shelf_id,l.id AS location_id,i.quantity
        FROM products p LEFT JOIN inventory_locations l ON l.business_id=p.business_id AND l.name=p.shelf_id
        LEFT JOIN inventory_counts i ON i.business_id=p.business_id AND i.product_id=p.id AND i.location_id=l.id
        WHERE p.business_id=?''',(shop,))}
    result=[]
    for sku,quantity in counts.items():
        db.validate_quantity(quantity)
        if sku not in products:
            raise PermissionError('Unknown current-shop SKU.')
        product=products[sku]
        if product['location_id'] is None:
            raise ValueError(f'Assign a valid shelf to {sku} before this action.')
        before=product['quantity'] if product['quantity'] is not None else 0
        after=before+quantity if operation=='add' else before-quantity
        if after<0:
            raise ValueError(f"{sku}: shelf {product['shelf_id']} has {before}; requested removal {quantity}. No changes saved.")
        result.append(dict(sku=sku,shelf_id=product['shelf_id'],location_id=product['location_id'],
                           before=before,reviewed_units=quantity,after=after))
    return result


def preview(service, counts, operation):
    user=service.user()
    with db.connect() as connection:
        return _preview(connection,user['business_id'],counts,operation)


def approve(service, photo_hash, submission_key, counts, predicted=None, notes='', operation='subtract', expected_preview=None):
    user=service.user()
    if not photo_hash or not submission_key or not counts or operation not in ('add','subtract'):
        raise ValueError('A reviewed photo, stock action and submission identity are required.')
    for quantity in counts.values():db.validate_quantity(quantity)
    prefix=submission_key+':shelf:'
    with db.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        previous=db.rows(connection.execute('''SELECT * FROM scan_sessions WHERE business_id=?
            AND (submission_key=? OR substr(submission_key,1,?)=?) ORDER BY id''',
            (user['business_id'],submission_key,len(prefix),prefix)))
        if previous:
            posted={}
            for receipt in previous:
                if (receipt['user_id']!=user['id'] or receipt['photo_hash']!=photo_hash
                        or receipt['operation']!=operation or receipt['scope']!='partial'):
                    raise ValueError('This scan was already approved with a different action.')
                for row in connection.execute('''SELECT p.sku,i.approved_quantity FROM scan_items i
                    JOIN products p ON p.id=i.product_id WHERE i.session_id=?''',(receipt['id'],)):
                    if row['sku'] in posted:raise ValueError('Conflicting saved scan receipts.')
                    posted[row['sku']]=row['approved_quantity']
            if posted!=counts:raise ValueError('This scan was already approved with different quantities.')
            return [r['id'] for r in previous]
        current=_preview(connection,user['business_id'],counts,operation)
        if expected_preview is not None:
            ordered=lambda rows:sorted(rows,key=lambda row:row['sku'])
            if ordered(current)!=ordered(expected_preview):
                raise ValueError('Stock or shelf assignments changed. Refresh the preview and confirm again.')
        groups={}
        for row in current:
            groups.setdefault(row['location_id'],{})[row['sku']]=row['reviewed_units']
        receipts=[]
        for location,group in sorted(groups.items()):
            key=submission_key if not receipts else prefix+str(location)
            receipts.append(db.create_scan_session(user['business_id'],user['id'],location,photo_hash,group,
                {sku:(predicted or {}).get(sku,0) for sku in group},notes,key,'partial',operation,_connection=connection))
        return receipts
