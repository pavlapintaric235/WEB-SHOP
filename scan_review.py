"""Draft corrections independent of inference and the stock ledger."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import db
from live_scan import valid_box, review_counts


def new_draft(items):
    return {'rows': [dict(item_id=f"D{item.get('item_id') or i}", sku=item.get('sku') or '', quantity=1,
                         confirmed=False, exclude=False, origin='detector', parent_id=None,
                         box_source='detector', suggestions=deepcopy(item.get('ranked_scores', [])),
                         **dict(zip(('x1','y1','x2','y2'), item['box'])))
                     for i, item in enumerate(items, 1)], 'events': [], 'next_id': 1, 'revision': 0}


def event(draft, action, actor, **details):
    draft['events'].append(dict(action=action, actor=actor, at_utc=datetime.now(timezone.utc).isoformat(), **details))
    draft['revision'] += 1


def add_manual(draft, sku, quantity, size, actor, parent_id=None, box=None):
    db.validate_quantity(quantity)
    if not quantity or not sku:
        raise ValueError('Choose an existing product and positive quantity.')
    coords = valid_box(box or [0, 0, *size], size)
    identity = f"M{draft['next_id']}"
    draft['next_id'] += 1
    row = dict(item_id=identity, sku=sku, quantity=quantity, confirmed=True, exclude=False,
               origin='manual', parent_id=parent_id, box_source='manual' if box else 'unlocalized',
               **dict(zip(('x1','y1','x2','y2'), coords)))
    draft['rows'].append(row)
    event(draft, 'manual_add', actor, item_id=identity, sku=sku, quantity=quantity, parent_id=parent_id)
    return identity


def edit_item(draft, identity, changes, actor):
    row = next(r for r in draft['rows'] if r['item_id'] == identity)
    allowed = {'sku', 'quantity', 'confirmed', 'exclude', 'x1','y1','x2','y2'}
    if set(changes) - allowed:
        raise ValueError('Unsupported detection edit.')
    before = deepcopy(row)
    row.update(changes)
    row['revision'] = row.get('revision', 0) + 1
    if row['box_source'] == 'unlocalized' and any(row[k] != before[k] for k in ('x1','y1','x2','y2')):
        row['box_source'] = 'manual'
    event(draft, 'item_edit', actor, item_id=identity, before=before, after=deepcopy(row))


def set_quantity(draft, sku, quantity, size, actor):
    db.validate_quantity(quantity)
    active = [r for r in draft['rows'] if r['sku'] == sku and not r['exclude']]
    current = sum(r['quantity'] for r in active)
    if quantity > current:
        add_manual(draft, sku, quantity-current, size, actor)
    elif quantity < current:
        remainder = current - quantity
        for row in reversed(active):
            take = min(remainder, row['quantity'])
            changes = {'exclude': True} if take == row['quantity'] else {'quantity': row['quantity']-take}
            edit_item(draft, row['item_id'], changes, actor)
            remainder -= take
            if not remainder:
                break


def split_item(draft, identity, children, size, actor):
    parent = next(r for r in draft['rows'] if r['item_id'] == identity)
    if parent['exclude'] or not 2 <= len(children) <= 8:
        raise ValueError('Split an active box into 2–8 instances.')
    checked = [(c['sku'], valid_box(c['box'], size)) for c in children]
    if any(not sku for sku, _ in checked):
        raise ValueError('Select a known product for every split instance.')
    edit_item(draft, identity, {'exclude': True}, actor)
    for sku, box in checked:
        child = add_manual(draft, sku, 1, size, actor, parent_id=identity, box=box)
        next(r for r in draft['rows'] if r['item_id'] == child)['confirmed'] = False
    event(draft, 'split', actor, parent_id=identity, children=len(children))


def grouped(draft):
    groups = {}
    for row in draft['rows']:
        if row['exclude']:
            continue
        group = groups.setdefault(row['sku'], {'quantity': 0, 'reviewed': True, 'rows': []})
        group['quantity'] += row['quantity']
        group['reviewed'] = group['reviewed'] and row['confirmed'] and bool(row['sku'])
        group['rows'].append(row)
    return groups


def reviewed_counts(draft, allowed, size, allow_empty=False):
    return review_counts(draft['rows'], set(allowed), size, allow_empty=allow_empty)


def stock_preview(inventory, counts, operation):
    if operation not in {'audit', 'add', 'subtract'}:
        raise ValueError('Choose an inventory action.')
    by_sku = {r['sku']: r for r in inventory}
    preview = []
    for sku, units in counts.items():
        db.validate_quantity(units)
        if sku not in by_sku:
            raise ValueError('Unknown current-shop SKU.')
        before = by_sku[sku]['quantity'] or 0
        after = units if operation == 'audit' else before + units * (1 if operation == 'add' else -1)
        if after < 0:
            raise ValueError(f'Removal exceeds stock for {sku}.')
        preview.append(dict(sku=sku, before=before, reviewed_units=units, after=after))
    return preview


def price_estimate(products, quantities):
    catalog = {p['sku']: p for p in products}
    total, lines, excluded = Decimal('0.00'), [], []
    for sku, units in quantities.items():
        db.validate_quantity(units)
        if not units:
            continue
        product = catalog.get(sku)
        price = product.get('price') if product else None
        if price is None:
            excluded.append(sku or 'Unknown product')
            lines.append(dict(sku=sku, quantity=units, price=None, subtotal=None))
            continue
        amount = Decimal(str(price))
        if not amount.is_finite() or amount < 0:
            raise ValueError('Invalid catalogue price.')
        subtotal = (amount * units).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
        total += subtotal
        lines.append(dict(sku=sku, name=product['name'], quantity=units, price=float(amount), subtotal=float(subtotal)))
    return dict(total=float(total), complete=not excluded, excluded=excluded, lines=lines)
