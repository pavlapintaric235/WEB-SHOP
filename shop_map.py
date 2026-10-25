"""Owner-entered grid, shelf highlights and explicitly manual visit progress."""
from collections import deque
from html import escape
import hashlib
import json

import pandas as pd
import streamlit as st

import db
from shop_catalog import validate_shelves


def load_layout(business_id):
    with db.connect() as connection:
        record = connection.execute('SELECT layout_json FROM shop_layouts WHERE business_id=?', (business_id,)).fetchone()
        if record:
            return json.loads(record['layout_json'])
        shelves = db.rows(connection.execute('SELECT shelf_id,zone,aisle,map_x,map_y FROM shop_shelves WHERE business_id=? ORDER BY shelf_id', (business_id,)))
    return dict(grid='', shelves=shelves)


def valid_grid(layout):
    grid = layout.get('grid', '').strip().splitlines()
    if not grid or not 1 <= len(grid) <= 40 or not 1 <= len(grid[0]) <= 40 or any(len(r) != len(grid[0]) for r in grid):
        raise ValueError('Provide a rectangular grid, at most 40 × 40 cells.')
    if any(c not in '.#EC' for row in grid for c in row) or sum(r.count('E') for r in grid) != 1 or sum(r.count('C') for r in grid) != 1:
        raise ValueError('Use . for walkable, # for blocked, exactly one E entrance and one C checkout.')
    for shelf in validate_shelves(layout['shelves']):
        x, y = shelf['map_x'], shelf['map_y']
        if x is None or y is None or y >= len(grid) or x >= len(grid[0]) or grid[y][x] != '#':
            raise ValueError(f"Place shelf {shelf['shelf_id']} on an in-bounds blocked (#) cell.")
    return grid


def _path(grid, start, targets):
    queue, previous = deque([start]), {start: None}
    while queue:
        cell = queue.popleft()
        if cell in targets:
            path = []
            while cell is not None:
                path.append(cell); cell = previous[cell]
            return list(reversed(path))
        x, y = cell
        for neighbor in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            nx, ny = neighbor
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != '#' and neighbor not in previous:
                previous[neighbor] = cell; queue.append(neighbor)
    return None


def visit_plan(layout, selected):
    ordered = sorted(set(selected))
    fallback = dict(order=ordered, path=[], routed=False)
    try:
        grid = valid_grid(layout)
    except ValueError:
        return fallback
    shelves = {s['shelf_id']: s for s in layout['shelves']}
    if not set(ordered) <= set(shelves):
        return fallback
    start = next((x,y) for y,row in enumerate(grid) for x,c in enumerate(row) if c == 'E')
    checkout = next((x,y) for y,row in enumerate(grid) for x,c in enumerate(row) if c == 'C')
    route, order, pending = [start], [], set(ordered)
    while pending:
        candidates = []
        for identity in sorted(pending):
            shelf = shelves[identity]; x,y = int(shelf['map_x']),int(shelf['map_y'])
            path = _path(grid, route[-1], {(x-1,y),(x+1,y),(x,y-1),(x,y+1)})
            if path:
                candidates.append((len(path), identity, path))
        if not candidates:
            return fallback
        _, identity, path = min(candidates)
        order.append(identity); pending.remove(identity); route.extend(path[1:])
    end = _path(grid, route[-1], {checkout})
    if end is None:
        return fallback
    route.extend(end[1:])
    return dict(order=order, path=route, routed=True)


def draw_map(layout, selected, path=(), compact=False):
    shelves = [s for s in layout['shelves'] if s.get('map_x') is not None and s.get('map_y') is not None]
    grid = layout.get('grid', '').strip().splitlines()
    width = max([len(grid[0]) if grid else 1] + [int(s['map_x'])+1 for s in shelves])
    height = max([len(grid) or 1] + [int(s['map_y'])+1 for s in shelves])
    size = 38
    view_box=f'0 0 {width*size} {height*size}'
    if compact and not grid and shelves:
        left=max(0,min(int(s['map_x']) for s in shelves)-1)
        top=max(0,min(int(s['map_y']) for s in shelves)-1)
        view_box=f'{left*size} {top*size} {(width-left)*size} {(height-top)*size}'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" role="img" aria-label="Owner shop layout">']
    for y in range(height):
        for x in range(width):
            c = grid[y][x] if grid and y < len(grid) and x < len(grid[y]) else '.'
            color = '#cbd5e1' if c == '#' else '#f8fafc'
            svg.append(f'<rect x="{x*size}" y="{y*size}" width="38" height="38" fill="{color}" stroke="#94a3b8"/>')
            if c in 'EC':
                svg.append(f'<text x="{x*size+8}" y="{y*size+24}" font-size="13">{c}</text>')
    if path:
        points = ' '.join(f'{x*size+19},{y*size+19}' for x,y in path)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    for shelf in shelves:
        x,y = int(shelf['map_x'])*size,int(shelf['map_y'])*size
        color = '#fbbf24' if shelf['shelf_id'] in selected else '#86efac'
        svg.append(f'<rect x="{x}" y="{y}" width="38" height="38" fill="{color}" stroke="#334155"/><text x="{x+3}" y="{y+24}" font-size="11">{escape(shelf["shelf_id"])}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def owner_map(service):
    user = service.user(owner=True)
    layout = load_layout(user['business_id'])
    st.subheader('Owner shelf map')
    st.caption('Zero-based x/y cells. Positions are owner-entered. E = entrance, C = checkout, . = walkable aisle, # = shelf or blocked cell.')
    known = {l['name'] for l in service.locations()}
    known.update(p['shelf_id'] for p in service.products() if p.get('shelf_id'))
    rows = layout['shelves'] or [dict(shelf_id=s, zone='', aisle='', map_x=None, map_y=None) for s in sorted(known)]
    frame = st.data_editor(pd.DataFrame(rows, columns=['shelf_id','zone','aisle','map_x','map_y']),
        hide_index=True, width='stretch', num_rows='dynamic', key=f'layout_shelves_{user["business_id"]}')
    grid = st.text_area('Walkable grid (leave blank to save shelf positions only)', value=layout['grid'], key=f'layout_grid_{user["business_id"]}')
    if st.button('Save shelf layout'):
        try:
            records = frame.astype(object).where(frame.notna(), '').to_dict('records')
            shelves = validate_shelves(records)
            if any(r['shelf_id'] not in known for r in shelves):
                raise ValueError('Use existing owner shelf/location IDs. Add a location before positioning it.')
            updated = dict(grid=grid.strip(), shelves=shelves)
            if grid.strip():
                valid_grid(updated)
            with db.connect() as connection:
                connection.execute('INSERT INTO shop_layouts VALUES (?,?) ON CONFLICT(business_id) DO UPDATE SET layout_json=excluded.layout_json', (user['business_id'], json.dumps(updated)))
            st.success('Shop layout saved.')
            layout = updated
        except (ValueError, TypeError) as error:
            st.error(str(error))
    st.image(draw_map(layout, []), width='stretch')


def buyer_map(business_id, selected, show_empty=False):
    st.subheader('Shopping stops')
    layout = load_layout(business_id)
    plan = visit_plan(layout, selected)
    if not selected and not show_empty:
        st.caption('Add products with owner-entered shelf IDs to see stops.')
        return
    if show_empty and not layout['shelves']:
        st.info('No shelf positions are saved yet. The owner can set them in Owner → Shelves.')
        return
    st.image(draw_map(layout, selected, plan['path'],compact=show_empty), width='stretch')
    if show_empty:st.caption('Yellow: shelves on your list. Green: other shelves. Owner → Shelves edits saved positions.')
    if not selected:
        st.caption('Add products with assigned shelves to your shopping list to highlight their locations.')
        return
    st.caption('Nearest-next-shelf visit order with BFS paths through owner walkable cells. This is not an optimal whole-store tour.' if plan['routed'] else 'Ordered shelf list only. No valid connected walkable grid is available for these stops.')
    stops = plan['order'] + (['Checkout'] if plan['routed'] else [])
    signature = hashlib.sha256(json.dumps(stops).encode()).hexdigest()[:12]
    key = f'map_stop_{business_id}_{signature}'
    index = st.session_state.setdefault(key, 0)
    st.write(' → '.join(stops))
    if index < len(stops):
        st.info(f'Next stop: {stops[index]}')
        if st.button('I reached this stop — next', key=f'{key}_next'):
            st.session_state[key] += 1; st.rerun()
    else:
        st.success('All selected stops visited.')
    st.caption('Advance manually. The app does not track your indoor position.')
