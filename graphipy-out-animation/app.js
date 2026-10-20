/* Web&Shop / InventoryLens living map: intro animation + interactive 3D explorer.
   Plain script (no modules) so the page also works when opened from file://. */
(function () {
  'use strict';

  var D = window.GRAPH_DATA, S = window.STORY, THREE = window.THREE;
  if (!D || !S || !THREE || !THREE.OrbitControls) {
    document.body.innerHTML = '<p style="padding:40px;font-family:sans-serif">Missing graph-data.js, story.js or the vendor libraries.</p>';
    return;
  }

  // ------------------------------------------------------------------ helpers
  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function fmt(n) { return Number(n).toLocaleString('en-US'); }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function lerp(a, b, k) { return a + (b - a) * k; }
  var ease = {
    linear: function (t) { return t; },
    outCubic: function (t) { return 1 - Math.pow(1 - t, 3); },
    inOutCubic: function (t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; },
    outQuint: function (t) { return 1 - Math.pow(1 - t, 5); },
    outBack: function (t) { var c = 1.70158; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); }
  };

  var tweens = [];
  function tween(dur, update, opts) {
    opts = opts || {};
    var tw = { t0: performance.now() + (opts.delay || 0), dur: dur, update: update,
      ease: opts.ease || ease.inOutCubic, done: opts.done, tag: opts.tag || '' };
    tweens.push(tw);
    return tw;
  }
  function stepTweens(now) {
    for (var i = tweens.length - 1; i >= 0; i--) {
      var tw = tweens[i];
      if (now < tw.t0) continue;
      var k = clamp((now - tw.t0) / tw.dur, 0, 1);
      tw.update(tw.ease(k), k);
      if (k >= 1) { tweens.splice(i, 1); if (tw.done) tw.done(); }
    }
  }
  function killTweens(tag) {
    for (var i = tweens.length - 1; i >= 0; i--) if (!tag || tweens[i].tag === tag) tweens.splice(i, 1);
  }

  var toastTimer = null;
  function toast(msg) {
    var el = $('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.classList.remove('show'); }, 2600);
  }

  // ------------------------------------------------------------------ data prep
  var nodes = D.nodes, links = D.links, aggLinks = D.aggLinks || [], comms = D.communities;
  var N = nodes.length, L = links.length, A = aggLinks.length;
  var idIndex = {};
  nodes.forEach(function (n, i) { n.i = i; idIndex[n.id] = i; });
  var commById = {};
  comms.forEach(function (c) { commById[c.id] = c; c.members = []; });
  nodes.forEach(function (n) { commById[n.c].members.push(n.i); });

  var PALETTE = ['#4de3ff', '#ff5fd2', '#a6ff4d', '#ffc857', '#8b7dff', '#3ddc97', '#ff8c5a', '#59c8ff',
    '#f75590', '#c8ff7a', '#ffd166', '#7cffcb', '#ff9de2', '#8ecae6', '#b5ff5b', '#ffb347', '#9d7bff',
    '#5fffe1', '#ff6b6b', '#c3f0ff', '#e0aaff', '#ffe066', '#80ffdb', '#ffa5e0', '#7fd1ff', '#d4ff7a'];
  (function assignColors() {
    var used = {};
    Object.keys(S.communityColors).forEach(function (k) {
      if (commById[k]) { commById[k].color = S.communityColors[k]; used[S.communityColors[k].toLowerCase()] = 1; }
    });
    var ranked = comms.slice().sort(function (a, b) { return b.size - a.size; });
    var pi = 0;
    ranked.forEach(function (c, r) {
      if (c.color) return;
      while (pi < PALETTE.length && used[PALETTE[pi].toLowerCase()]) pi++;
      if (pi < PALETTE.length) { c.color = PALETTE[pi++]; }
      else {
        var col = new THREE.Color().setHSL(((r * 137.508) % 360) / 360, 0.78, 0.64);
        c.color = '#' + col.getHexString();
      }
    });
    comms.forEach(function (c) { c.col = new THREE.Color(c.color); c.rank = ranked.indexOf(c); });
  })();

  var DOC_KINDS = { concept: 1, document: 1, image: 1, rationale: 1 };
  var OVERVIEW_KINDS = { file: 1, document: 1, image: 1, concept: 1 };
  nodes.forEach(function (n) {
    n.col = commById[n.c].col;
    n.rad = 1.15 + 1.45 * Math.log2(1 + n.d);
    n.doc = !!DOC_KINDS[n.k];
    n.file = n.f >= 0 ? D.files[n.f] : '';
  });

  var adj = new Array(N);
  for (var i = 0; i < N; i++) adj[i] = [];
  links.forEach(function (l, li) { adj[l[0]].push(li); adj[l[1]].push(li); });
  var aggAdj = new Array(N);
  aggLinks.forEach(function (l, ai) {
    (aggAdj[l[0]] = aggAdj[l[0]] || []).push(ai);
    (aggAdj[l[1]] = aggAdj[l[1]] || []).push(ai);
  });

  var blocks = S.blocks;
  var blockByKey = {};
  blocks.forEach(function (b) {
    blockByKey[b.key] = b;
    b.anchorIdx = idIndex[b.anchor] != null ? idIndex[b.anchor] : -1;
    b.col = new THREE.Color(b.color);
  });

  // ------------------------------------------------------------------ renderer
  var canvas = $('gl');
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setClearColor(0x000000, 1);
  var scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x000000, 0.00038);
  var camera = new THREE.PerspectiveCamera(55, 1, 1, 30000);
  scene.add(camera);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  var headlight = new THREE.DirectionalLight(0xffffff, 0.85);
  headlight.position.set(0.5, 0.8, 1);
  camera.add(headlight);

  var controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.7;
  controls.minDistance = 25;
  controls.maxDistance = 8000;
  controls.autoRotateSpeed = 0.45;
  controls.enabled = false;

  var graphGroup = new THREE.Group();
  scene.add(graphGroup);

  // Nodes: spheres for code, octahedra for documentation-derived items.
  var nodeMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 1 });
  var sphereGeo = new THREE.SphereGeometry(1, 12, 9);
  var octaGeo = new THREE.OctahedronGeometry(1.3, 0);
  var codeIdx = [], docIdx = [];
  nodes.forEach(function (n) { (n.doc ? docIdx : codeIdx).push(n.i); });
  var codeMesh = new THREE.InstancedMesh(sphereGeo, nodeMat, Math.max(1, codeIdx.length));
  var docMesh = new THREE.InstancedMesh(octaGeo, nodeMat, Math.max(1, docIdx.length));
  codeMesh.frustumCulled = docMesh.frustumCulled = false;
  codeMesh.userData.map = codeIdx;
  docMesh.userData.map = docIdx;
  codeIdx.forEach(function (ni, slot) { nodes[ni].mesh = codeMesh; nodes[ni].slot = slot; });
  docIdx.forEach(function (ni, slot) { nodes[ni].mesh = docMesh; nodes[ni].slot = slot; });
  graphGroup.add(codeMesh);
  graphGroup.add(docMesh);

  // Halo glow as a single Points draw call.
  var haloGeo = new THREE.BufferGeometry();
  var haloPos = new Float32Array(N * 3), haloCol = new Float32Array(N * 3), haloSize = new Float32Array(N), haloAlpha = new Float32Array(N);
  nodes.forEach(function (n) {
    haloPos[n.i * 3] = n.x; haloPos[n.i * 3 + 1] = n.y; haloPos[n.i * 3 + 2] = n.z;
    haloSize[n.i] = n.rad * 2.8;
  });
  haloGeo.setAttribute('position', new THREE.BufferAttribute(haloPos, 3));
  haloGeo.setAttribute('color', new THREE.BufferAttribute(haloCol, 3));
  haloGeo.setAttribute('size', new THREE.BufferAttribute(haloSize, 1));
  haloGeo.setAttribute('alpha', new THREE.BufferAttribute(haloAlpha, 1));
  var haloMat = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 300 }, uGlobal: { value: 1 } },
    vertexShader: [
      'attribute float size; attribute float alpha; attribute vec3 color;',
      'uniform float uScale; varying vec3 vColor; varying float vAlpha;',
      'void main(){ vColor = color; vAlpha = alpha;',
      '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
      '  gl_PointSize = clamp(size * 2.0 * uScale / max(1.0, -mv.z), 1.0, 420.0);',
      '  gl_Position = projectionMatrix * mv; }'
    ].join('\n'),
    fragmentShader: [
      'uniform float uGlobal; varying vec3 vColor; varying float vAlpha;',
      'void main(){ float d = length(gl_PointCoord - 0.5) * 2.0;',
      '  float a = smoothstep(1.0, 0.0, d); a = a * a * a;',
      '  gl_FragColor = vec4(vColor, a * vAlpha * uGlobal); }'
    ].join('\n'),
    transparent: true, depthWrite: false, depthTest: true, blending: THREE.AdditiveBlending
  });
  var halo = new THREE.Points(haloGeo, haloMat);
  halo.frustumCulled = false;
  halo.renderOrder = 2;
  graphGroup.add(halo);

  // Links as instanced thin tubes (one draw call), additive so overlaps glow.
  var cylGeo = new THREE.CylinderGeometry(1, 1, 1, 5, 1, true);
  cylGeo.translate(0, 0.5, 0);
  var linkMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false });
  var linkMesh = new THREE.InstancedMesh(cylGeo, linkMat, Math.max(1, L));
  var aggMesh = new THREE.InstancedMesh(cylGeo, linkMat, Math.max(1, A));
  linkMesh.frustumCulled = aggMesh.frustumCulled = false;
  graphGroup.add(linkMesh);
  graphGroup.add(aggMesh);

  var Y_AXIS = new THREE.Vector3(0, 1, 0);
  var _m = new THREE.Matrix4(), _v = new THREE.Vector3(), _v2 = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _c = new THREE.Color();
  function linkGeom(list) {
    return list.map(function (l) {
      var a = nodes[l[0]], b = nodes[l[1]];
      var dir = new THREE.Vector3(b.x - a.x, b.y - a.y, b.z - a.z);
      var len = Math.max(0.001, dir.length());
      var q = new THREE.Quaternion().setFromUnitVectors(Y_AXIS, dir.normalize());
      return { a: new THREE.Vector3(a.x, a.y, a.z), q: q, len: len, mix: a.col.clone().lerp(b.col, 0.5) };
    });
  }
  var linkG = linkGeom(links);
  var aggG = linkGeom(aggLinks);
  function setTube(mesh, slot, g, radius, color) {
    _s.set(radius, g.len, radius);
    _m.compose(g.a, g.q, _s);
    mesh.setMatrixAt(slot, _m);
    mesh.setColorAt(slot, color);
  }

  // Selection ring + sparks travelling along highlighted links.
  var ring = new THREE.Mesh(new THREE.RingGeometry(1.55, 1.75, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false }));
  ring.visible = false;
  ring.renderOrder = 5;
  scene.add(ring);

  var MAX_SPARKS = 240;
  var sparkGeo = new THREE.BufferGeometry();
  var sparkPos = new Float32Array(MAX_SPARKS * 3), sparkCol = new Float32Array(MAX_SPARKS * 3);
  sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
  sparkGeo.setAttribute('color', new THREE.BufferAttribute(sparkCol, 3));
  var sparkMat = new THREE.PointsMaterial({ size: 5.5, vertexColors: true, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true });
  var sparks = new THREE.Points(sparkGeo, sparkMat);
  sparks.frustumCulled = false;
  sparks.renderOrder = 3;
  graphGroup.add(sparks);
  var sparkLinks = [];
  sparkGeo.setDrawRange(0, 0);

  // ------------------------------------------------------------------ state
  var level = 'full';
  var isolateComm = null;
  var nodeVis = new Uint8Array(N), linkVis = new Uint8Array(L), aggVis = new Uint8Array(A);
  var sel = null;             // { type: 'node'|'link'|'agg'|'community', idx }
  var hi = null;              // { nodes: {}, links: {}, aggs: {} }
  var hoverNode = -1, hoverLink = -1, hoverAgg = -1;
  var labelsOn = true;
  var mode = 'boot';          // 'intro' | 'explore'
  var visibleCount = 0, visibleLinkCount = 0;

  function computeVisibility() {
    visibleCount = 0; visibleLinkCount = 0;
    for (var i = 0; i < N; i++) {
      var n = nodes[i];
      var v = isolateComm != null ? (n.c === isolateComm) : (level === 'full' || !!OVERVIEW_KINDS[n.k]);
      nodeVis[i] = v ? 1 : 0;
      if (v) visibleCount++;
    }
    for (var li = 0; li < L; li++) {
      linkVis[li] = (nodeVis[links[li][0]] && nodeVis[links[li][1]]) ? 1 : 0;
      if (linkVis[li]) visibleLinkCount++;
    }
    for (var ai = 0; ai < A; ai++) {
      aggVis[ai] = (level === 'overview' && isolateComm == null && nodeVis[aggLinks[ai][0]] && nodeVis[aggLinks[ai][1]]) ? 1 : 0;
      if (aggVis[ai]) visibleLinkCount++;
    }
    updateStatus();
  }

  function computeHighlight() {
    hi = null;
    if (!sel) return;
    var h = { nodes: {}, links: {}, aggs: {} };
    if (sel.type === 'node') {
      h.nodes[sel.idx] = 2;
      adj[sel.idx].forEach(function (li) {
        if (!linkVis[li]) return;
        h.links[li] = 1;
        var l = links[li];
        h.nodes[l[0] === sel.idx ? l[1] : l[0]] = h.nodes[l[0] === sel.idx ? l[1] : l[0]] || 1;
      });
      (aggAdj[sel.idx] || []).forEach(function (ai) {
        if (!aggVis[ai]) return;
        h.aggs[ai] = 1;
        var l = aggLinks[ai];
        h.nodes[l[0] === sel.idx ? l[1] : l[0]] = h.nodes[l[0] === sel.idx ? l[1] : l[0]] || 1;
      });
    } else if (sel.type === 'link') {
      var l = links[sel.idx];
      h.nodes[l[0]] = 2; h.nodes[l[1]] = 2; h.links[sel.idx] = 2;
    } else if (sel.type === 'agg') {
      var g = aggLinks[sel.idx];
      h.nodes[g[0]] = 2; h.nodes[g[1]] = 2; h.aggs[sel.idx] = 2;
    } else if (sel.type === 'community') {
      var c = commById[sel.idx];
      c.members.forEach(function (i) { h.nodes[i] = 1; });
      for (var li = 0; li < L; li++) if (linkVis[li] && h.nodes[links[li][0]] && h.nodes[links[li][1]]) h.links[li] = 1;
      for (var ai = 0; ai < A; ai++) if (aggVis[ai] && h.nodes[aggLinks[ai][0]] && h.nodes[aggLinks[ai][1]]) h.aggs[ai] = 1;
    }
    hi = h;
  }

  var DIM_NODE = 0.14, DIM_LINK = 0.05, WHITE = new THREE.Color(0xffffff);
  function applyStyles() {
    var hasHi = !!hi;
    for (var i = 0; i < N; i++) {
      var n = nodes[i];
      var scale, alpha;
      if (!nodeVis[i]) { scale = 0.0001; alpha = 0; _c.setRGB(0, 0, 0); }
      else if (!hasHi) { scale = n.rad; alpha = 0.42; _c.copy(n.col); }
      else if (hi.nodes[i] === 2) { scale = n.rad * 1.2; alpha = 0.95; _c.copy(n.col).lerp(WHITE, 0.45); }
      else if (hi.nodes[i]) { scale = n.rad * 1.08; alpha = 0.8; _c.copy(n.col).lerp(WHITE, 0.15); }
      else { scale = n.rad * 0.85; alpha = 0.04; _c.copy(n.col).multiplyScalar(DIM_NODE); }
      if (hoverNode === i && nodeVis[i]) { scale *= 1.2; alpha = Math.max(alpha, 0.85); _c.copy(n.col).lerp(WHITE, 0.35); }
      _m.makeScale(scale, scale, scale);
      _m.setPosition(n.x, n.y, n.z);
      n.mesh.setMatrixAt(n.slot, _m);
      n.mesh.setColorAt(n.slot, _c);
      haloCol[i * 3] = _c.r; haloCol[i * 3 + 1] = _c.g; haloCol[i * 3 + 2] = _c.b;
      haloAlpha[i] = alpha;
    }
    codeMesh.instanceMatrix.needsUpdate = docMesh.instanceMatrix.needsUpdate = true;
    if (codeMesh.instanceColor) codeMesh.instanceColor.needsUpdate = true;
    if (docMesh.instanceColor) docMesh.instanceColor.needsUpdate = true;
    haloGeo.attributes.color.needsUpdate = haloGeo.attributes.alpha.needsUpdate = true;

    var baseR = level === 'full' ? 0.26 : 0.4;
    for (var li = 0; li < L; li++) {
      var g = linkG[li];
      if (!linkVis[li]) { _c.setRGB(0, 0, 0); setTube(linkMesh, li, g, 0.0001, _c); continue; }
      var r = baseR;
      if (!hasHi) _c.copy(g.mix).multiplyScalar(0.55);
      else if (hi.links[li] === 2) { _c.copy(g.mix).lerp(WHITE, 0.5); r = baseR * 1.9; }
      else if (hi.links[li]) { _c.copy(g.mix).lerp(WHITE, 0.25); r = baseR * 1.5; }
      else { _c.copy(g.mix).multiplyScalar(DIM_LINK); r = baseR * 0.8; }
      if (hoverLink === li) { _c.copy(g.mix).lerp(WHITE, 0.6); r = baseR * 2.1; }
      setTube(linkMesh, li, g, r, _c);
    }
    linkMesh.instanceMatrix.needsUpdate = true;
    if (linkMesh.instanceColor) linkMesh.instanceColor.needsUpdate = true;

    for (var ai = 0; ai < A; ai++) {
      var ag = aggG[ai], al = aggLinks[ai];
      if (!aggVis[ai]) { _c.setRGB(0, 0, 0); setTube(aggMesh, ai, ag, 0.0001, _c); continue; }
      var ar = 0.55 + 0.45 * Math.log2(1 + al[2]);
      if (!hasHi) _c.copy(ag.mix).multiplyScalar(0.5);
      else if (hi.aggs[ai] === 2) { _c.copy(ag.mix).lerp(WHITE, 0.5); ar *= 1.5; }
      else if (hi.aggs[ai]) { _c.copy(ag.mix).lerp(WHITE, 0.25); ar *= 1.25; }
      else { _c.copy(ag.mix).multiplyScalar(DIM_LINK); }
      if (hoverAgg === ai) { _c.copy(ag.mix).lerp(WHITE, 0.6); ar *= 1.6; }
      setTube(aggMesh, ai, ag, ar, _c);
    }
    aggMesh.instanceMatrix.needsUpdate = true;
    if (aggMesh.instanceColor) aggMesh.instanceColor.needsUpdate = true;

    rebuildSparks();
  }

  function rebuildSparks() {
    sparkLinks = [];
    if (hi) {
      Object.keys(hi.links).forEach(function (li) { if (sparkLinks.length < MAX_SPARKS) sparkLinks.push({ g: linkG[li], phase: Math.random(), col: linkG[li].mix }); });
      Object.keys(hi.aggs).forEach(function (ai) { if (sparkLinks.length < MAX_SPARKS) sparkLinks.push({ g: aggG[ai], phase: Math.random(), col: aggG[ai].mix }); });
    }
    sparkLinks.forEach(function (sp, i) {
      _c.copy(sp.col).lerp(new THREE.Color(0xffffff), 0.5);
      sparkCol[i * 3] = _c.r; sparkCol[i * 3 + 1] = _c.g; sparkCol[i * 3 + 2] = _c.b;
    });
    sparkGeo.attributes.color.needsUpdate = true;
    sparkGeo.setDrawRange(0, sparkLinks.length);
  }
  function updateSparks(now) {
    if (!sparkLinks.length) return;
    for (var i = 0; i < sparkLinks.length; i++) {
      var sp = sparkLinks[i];
      var t = ((now / 1500) + sp.phase) % 1;
      _v.set(0, sp.g.len * t, 0).applyQuaternion(sp.g.q).add(sp.g.a);
      sparkPos[i * 3] = _v.x; sparkPos[i * 3 + 1] = _v.y; sparkPos[i * 3 + 2] = _v.z;
    }
    sparkGeo.attributes.position.needsUpdate = true;
  }

  function restyle() { computeHighlight(); applyStyles(); }

  // ------------------------------------------------------------------ camera helpers
  function worldOf(n, out) {
    out = out || new THREE.Vector3();
    out.set(n.x, n.y, n.z);
    return graphGroup.localToWorld(out);
  }
  function flyTo(target, distance, dur) {
    dur = dur || 1100;
    var p0 = camera.position.clone(), t0 = controls.target.clone();
    var dir = p0.clone().sub(t0);
    if (dir.length() < 1) dir.set(0, 0.3, 1);
    dir.normalize();
    var p1 = target.clone().add(dir.multiplyScalar(distance));
    killTweens('cam');
    tween(dur, function (k) {
      camera.position.lerpVectors(p0, p1, k);
      controls.target.lerpVectors(t0, target, k);
    }, { tag: 'cam', ease: ease.inOutCubic });
  }
  // Camera placement that frames every visible node inside the area left of the panel.
  function computeFit(useWorld, dir) {
    var box = new THREE.Box3();
    var any = false;
    for (var i = 0; i < N; i++) {
      if (!nodeVis[i]) continue;
      any = true;
      if (useWorld) worldOf(nodes[i], _v); else _v.set(nodes[i].x, nodes[i].y, nodes[i].z);
      box.expandByPoint(_v);
    }
    if (!any) return null;
    var sphere = box.getBoundingSphere(new THREE.Sphere());
    var W = window.innerWidth, H = window.innerHeight;
    var panelW = Math.min(W * 0.45, parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--panel-w')) || 440);
    var free = Math.max(0.4, (W - panelW) / W);
    var tanV = Math.tan(camera.fov * Math.PI / 360);
    var distV = sphere.radius / Math.sin(camera.fov * Math.PI / 360);
    var distH = sphere.radius / (tanV * (W / H) * free);
    var dist = Math.max(distV, distH) * 0.86 + 30;
    dir = (dir || new THREE.Vector3(0.18, 0.22, 1)).clone().normalize();
    var right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
    var worldPerPx = 2 * dist * tanV / H;
    var target = sphere.center.clone().add(right.multiplyScalar(panelW / 2 * worldPerPx));
    return { target: target, pos: target.clone().add(dir.multiplyScalar(dist)) };
  }
  function fitVisible(dur) {
    var dir = camera.position.clone().sub(controls.target);
    if (dir.length() < 1) dir.set(0.18, 0.22, 1);
    var fit = computeFit(true, dir);
    if (!fit) return;
    var p0 = camera.position.clone(), t0 = controls.target.clone();
    killTweens('cam');
    tween(dur || 1200, function (k) {
      camera.position.lerpVectors(p0, fit.pos, k);
      controls.target.lerpVectors(t0, fit.target, k);
    }, { tag: 'cam' });
  }

  // ------------------------------------------------------------------ labels
  var labelLayer = $('labels');
  var labelPool = [];
  var widthCache = {};
  function labelDiv(i) {
    while (labelPool.length <= i) {
      var d = document.createElement('div');
      d.className = 'lbl';
      labelLayer.appendChild(d);
      labelPool.push(d);
    }
    return labelPool[i];
  }
  var placed = [];
  function overlaps(r) {
    for (var i = 0; i < placed.length; i++) {
      var p = placed[i];
      if (r.x < p.x + p.w && r.x + r.w > p.x && r.y < p.y + p.h && r.y + r.h > p.y) return true;
    }
    return false;
  }
  function estWidth(text, cls) {
    var key = cls + '|' + text;
    if (widthCache[key]) return widthCache[key];
    var per = cls === 'node' ? 7.4 : cls === 'comm' ? 9.2 : cls === 'block' ? 12.2 : 9.6;
    var w = text.length * per + 22;
    widthCache[key] = w;
    return w;
  }
  var introLabels = [];  // filled by the intro with {obj, text, color}
  function updateLabels() {
    var W = canvas.clientWidth, H = canvas.clientHeight;
    var cands = [];
    var camPos = camera.position;
    function add(pos, text, cls, color, pri, extraCls) {
      _v.copy(pos).project(camera);
      if (_v.z > 1 || _v.z < -1) return;
      var sx = (_v.x + 1) / 2 * W, sy = (1 - _v.y) / 2 * H;
      if (sx < -200 || sx > W + 200 || sy < -50 || sy > H + 50) return;
      cands.push({ x: sx, y: sy, text: text, cls: cls, color: color, pri: pri, extra: extraCls || '', dist: pos.distanceTo(camPos) });
    }
    if (mode === 'intro') {
      introLabels.forEach(function (il) {
        if (!il.obj.visible || il.obj.scale.x < 0.05) return;
        il.obj.getWorldPosition(_v2);
        _v2.y += 26 * il.obj.scale.x;
        add(_v2, il.text, 'block', il.color, 1000, '');
      });
    } else if (labelsOn) {
      var hasHi = !!hi;
      // Tour blocks keep their names as beacons.
      blocks.forEach(function (b) {
        if (b.anchorIdx < 0 || !nodeVis[b.anchorIdx]) return;
        if (hasHi && !hi.nodes[b.anchorIdx]) return;
        var n = nodes[b.anchorIdx];
        worldOf(n, _v2); _v2.y += n.rad * 1.6 + 2;
        add(_v2, b.name, 'beacon', b.color, 900 + n.d, '');
      });
      // Community (block) names.
      comms.forEach(function (c) {
        if (isolateComm != null && c.id !== isolateComm) return;
        if (hasHi && sel.type !== 'community' && c.rank > 5 && !(sel.type === 'node' && nodes[sel.idx].c === c.id)) return;
        if (level === 'overview' && !c.members.some(function (i) { return nodeVis[i]; })) return;
        _v2.set(c.cx, c.cy + c.r * 0.9, c.cz);
        graphGroup.localToWorld(_v2);
        var d = _v2.distanceTo(camPos);
        if (c.rank > 34 && d > 700) return;
        add(_v2, c.name, 'comm', c.color, 500 + c.size, '');
      });
      // Node names: important nodes, the hovered node, the selection and its neighbours.
      var list = [];
      if (hasHi) Object.keys(hi.nodes).forEach(function (i) { list.push(+i); });
      else list = importantNodes;
      if (hoverNode >= 0 && list.indexOf(hoverNode) < 0) list = list.concat([hoverNode]);
      var maxLabels = hasHi ? 48 : 34;
      list.slice(0, 400).forEach(function (i) {
        var n = nodes[i];
        if (!nodeVis[i]) return;
        var pri = 100 + n.d;
        if (i === hoverNode) pri = 2000;
        if (sel && sel.type === 'node' && sel.idx === i) pri = 1900;
        worldOf(n, _v2); _v2.y -= n.rad * 1.4 + 1;
        add(_v2, n.l, 'node', null, pri, (i === hoverNode || (sel && sel.type === 'node' && sel.idx === i)) ? ' hot' : '');
      });
      cands.sort(function (a, b) { return b.pri - a.pri; });
      var kept = 0;
      cands = cands.filter(function (c) { if (c.cls !== 'node') return true; return kept++ < maxLabels; });
    }
    cands.sort(function (a, b) { return b.pri - a.pri; });
    placed.length = 0;
    var used = 0;
    for (var i = 0; i < cands.length && used < 140; i++) {
      var c = cands[i];
      var w = estWidth(c.text, c.cls), h = c.cls === 'block' ? 30 : c.cls === 'node' ? 22 : 24;
      var rect = { x: c.x - w / 2, y: c.cls === 'node' ? c.y : c.y - h, w: w, h: h };
      if (overlaps(rect)) continue;
      placed.push(rect);
      var div = labelDiv(used++);
      var cls = 'lbl ' + c.cls + c.extra;
      if (div.className !== cls) div.className = cls;
      if (div.textContent !== c.text) div.textContent = c.text;
      div.style.color = c.color || '';
      var op = c.cls === 'block' ? 1 : c.cls === 'node' ? clamp(1.25 - c.dist / 2600, 0.25, 1) : clamp(1.35 - c.dist / 3200, 0.3, 1);
      div.style.opacity = op;
      div.style.display = 'block';
      div.style.transform = 'translate(' + (c.x - w / 2).toFixed(1) + 'px,' + (c.cls === 'node' ? c.y : c.y - h).toFixed(1) + 'px)';
    }
    for (var j = used; j < labelPool.length; j++) if (labelPool[j].style.display !== 'none') labelPool[j].style.display = 'none';
  }
  var importantNodes = nodes.map(function (n) { return n.i; }).sort(function (a, b) { return nodes[b].d - nodes[a].d; }).slice(0, 220);

  // ------------------------------------------------------------------ picking
  var raycaster = new THREE.Raycaster();
  var ndc = new THREE.Vector2();
  var screenPos = new Float32Array(N * 3);
  function projectAll() {
    var W = canvas.clientWidth, H = canvas.clientHeight;
    for (var i = 0; i < N; i++) {
      if (!nodeVis[i]) { screenPos[i * 3 + 2] = 9; continue; }
      worldOf(nodes[i], _v).project(camera);
      screenPos[i * 3] = (_v.x + 1) / 2 * W;
      screenPos[i * 3 + 1] = (1 - _v.y) / 2 * H;
      screenPos[i * 3 + 2] = _v.z;
    }
  }
  function pickNode(px, py) {
    ndc.set(px / canvas.clientWidth * 2 - 1, -(py / canvas.clientHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    var hits = raycaster.intersectObjects([codeMesh, docMesh], false);
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i];
      if (h.instanceId == null) continue;
      var idx = h.object.userData.map[h.instanceId];
      if (idx != null && nodeVis[idx]) return idx;
    }
    return -1;
  }
  function segDist2(px, py, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    var l2 = dx * dx + dy * dy;
    var t = l2 ? clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0, 1) : 0;
    var x = ax + t * dx - px, y = ay + t * dy - py;
    return x * x + y * y;
  }
  function pickLink(px, py) {
    projectAll();
    var best = -1, bestType = '', bestD = 7 * 7;
    function test(list, vis, type) {
      for (var i = 0; i < list.length; i++) {
        if (!vis[i]) continue;
        var a = list[i][0], b = list[i][1];
        if (screenPos[a * 3 + 2] > 1 || screenPos[b * 3 + 2] > 1) continue;
        var d = segDist2(px, py, screenPos[a * 3], screenPos[a * 3 + 1], screenPos[b * 3], screenPos[b * 3 + 1]);
        if (d < bestD) { bestD = d; best = i; bestType = type; }
      }
    }
    test(links, linkVis, 'link');
    test(aggLinks, aggVis, 'agg');
    return best < 0 ? null : { type: bestType, idx: best };
  }

  // ------------------------------------------------------------------ panel
  var panelBody = $('panelBody');
  function commChip(cid) {
    var c = commById[cid];
    return '<span class="chip" data-comm="' + cid + '" style="color:' + c.color + '"><span class="dot"></span>' + esc(c.name) + '</span>';
  }
  function nodeLine(i, relText) {
    var n = nodes[i];
    return '<li data-node="' + i + '"><span class="sw" style="color:' + commById[n.c].color + '"></span>' +
      (relText ? '<span class="rel">' + esc(relText) + '</span>' : '') +
      '<span class="lbl-text">' + esc(n.l) + '</span><span class="deg">' + n.d + '</span></li>';
  }
  function relPhrase(rel, dirOut) {
    var map = {
      calls: ['calls', 'called by'], contains: ['contains', 'inside'], references: ['refers to', 'referred by'],
      imports: ['imports', 'imported by'], imports_from: ['imports from', 'exports to'], rationale_for: ['explains', 'explained by'],
      conceptually_related_to: ['related to', 'related to'], shares_data_with: ['shares data', 'shares data'],
      indirect_call: ['indirectly calls', 'indirectly called by'], method: ['has method', 'method of'], uses: ['uses', 'used by'],
      semantically_similar_to: ['similar to', 'similar to'], inherits: ['inherits from', 'parent of'],
      implements: ['implements', 'implemented by'], defines: ['defines', 'defined by']
    };
    var p = map[rel] || [rel.replace(/_/g, ' '), rel.replace(/_/g, ' ')];
    return dirOut ? p[0] : p[1];
  }

  function showHelp() {
    var h = '<div class="help"><div class="kicker">How to explore</div><h2>The living map</h2>' +
      '<p>Every dot is a piece of the Web&Shop / InventoryLens software: a file, a function, a rule or an idea from the documentation. ' +
      'Every line is a real connection between two of them. Colours mark the <b>blocks</b>: groups of things that work closely together.</p>' +
      '<ol>' +
      '<li><b>Rotate</b>: drag with the left mouse button.</li>' +
      '<li><b>Zoom</b>: scroll the mouse wheel. <b>Move</b>: drag with the right button.</li>' +
      '<li><b>Click</b> a dot or a line to read what it is, here in this panel. Double-click a dot to fly to it.</li>' +
      '<li><b>Overview</b> shows only files and documentation. <b>Everything</b> shows all ' + fmt(N) + ' pieces.</li>' +
      '<li>Type in the search box to find anything by name.</li>' +
      '</ol>' +
      '<p class="small muted"><span class="key">Esc</span> clear selection &nbsp; <span class="key">F</span> fit to view &nbsp; <span class="key">R</span> auto-rotate &nbsp; <span class="key">L</span> labels</p>' +
      '<h3>The nine blocks of the tour</h3><ul class="blocks">';
    blocks.forEach(function (b, i) {
      h += '<li data-node="' + b.anchorIdx + '" style="color:' + b.color + '"><span class="num">' + (i + 1) + '</span><span><div class="name">' + esc(b.name) + '</div><div class="desc">' + esc(b.short) + '</div></span></li>';
    });
    h += '</ul><h3>Biggest blocks</h3>';
    comms.slice().sort(function (a, b) { return b.size - a.size; }).slice(0, 12).forEach(function (c) { h += commChip(c.id); });
    h += '<h3>Group flows found in the documentation</h3><ul class="linklist">';
    (D.hyperedges || []).forEach(function (he, k) {
      h += '<li data-hyper="' + k + '"><span class="sw" style="color:#fff"></span><span class="lbl-text">' + esc(he.label) + '</span><span class="deg">' + he.nodes.length + '</span></li>';
    });
    h += '</ul></div>';
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
  }

  function showNode(i) {
    var n = nodes[i];
    var c = commById[n.c];
    var kindLabel = S.kindLabels[n.k] || n.k;
    var block = blocks.filter(function (b) { return b.anchorIdx === i; })[0];
    var h = '<div class="kicker">' + esc(kindLabel) + (n.o === 'semantic' ? ' · from documentation' : '') + '</div>';
    h += '<h2>' + esc(n.l) + '</h2>';
    h += commChip(n.c);
    if (block) h += '<span class="chip" style="color:' + block.color + '"><span class="dot"></span>Tour block ' + (blocks.indexOf(block) + 1) + ' · ' + esc(block.name) + '</span>';
    h += '<p class="muted">' + esc(S.kindNotes[n.k] || '') + '</p>';
    if (block) h += '<p>' + esc(block.short) + '. In the tour this stood for the whole block <b>' + esc(c.name) + '</b>.</p>';
    if (n.r) h += '<div class="quote">' + esc(n.r) + '</div>';
    h += '<dl class="fact">';
    if (n.file) h += '<dt>Where</dt><dd><span class="code">' + esc(n.file) + (n.loc ? ':' + esc(n.loc) : '') + '</span></dd>';
    else h += '<dt>Where</dt><dd class="muted">Outside this project (library or standard building block)</dd>';
    h += '<dt>Connections</dt><dd>' + fmt(n.d) + '</dd>';
    if (n.u) h += '<dt>Source</dt><dd><span class="code">' + esc(n.u) + '</span></dd>';
    h += '</dl>';
    h += '<div class="btnrow"><button class="btn" data-action="fly" data-node="' + i + '">Fly to</button>' +
      '<button class="btn" data-action="isolate" data-comm="' + n.c + '">Only this block</button></div>';

    var groups = {};
    adj[i].forEach(function (li) {
      var l = links[li];
      var out = l[0] === i;
      var other = out ? l[1] : l[0];
      var key = relPhrase(D.relations[l[2]], out);
      (groups[key] = groups[key] || []).push({ other: other, li: li, hidden: !linkVis[li] });
    });
    var keys = Object.keys(groups).sort(function (a, b) { return groups[b].length - groups[a].length; });
    if (keys.length) {
      h += '<h3>Connected to (' + fmt(adj[i].length) + ')</h3>';
      keys.forEach(function (k) {
        var items = groups[k].sort(function (a, b) { return nodes[b.other].d - nodes[a.other].d; });
        h += '<div class="kicker" style="margin:12px 0 4px;color:var(--fg-dim)">' + esc(k) + ' · ' + items.length + '</div><ul class="linklist">';
        items.slice(0, 30).forEach(function (it) {
          h += '<li data-link="' + it.li + '"><span class="sw" style="color:' + commById[nodes[it.other].c].color + '"></span><span class="lbl-text">' + esc(nodes[it.other].l) +
            (it.hidden ? ' <span class="muted small">(hidden at this detail level)</span>' : '') + '</span><span class="deg">' + nodes[it.other].d + '</span></li>';
        });
        if (items.length > 30) h += '<li class="muted small">… and ' + (items.length - 30) + ' more</li>';
        h += '</ul>';
      });
    }
    var aggs = aggAdj[i] || [];
    if (aggs.length && level === 'overview') {
      h += '<h3>File-level connections (' + aggs.length + ')</h3><ul class="linklist">';
      aggs.slice().sort(function (a, b) { return aggLinks[b][2] - aggLinks[a][2]; }).forEach(function (ai) {
        var g = aggLinks[ai], other = g[0] === i ? g[1] : g[0];
        h += '<li data-agg="' + ai + '"><span class="sw" style="color:' + commById[nodes[other].c].color + '"></span><span class="lbl-text">' + esc(nodes[other].l) + '</span><span class="deg">' + g[2] + '</span></li>';
      });
      h += '</ul>';
    }
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
  }

  function showLink(li) {
    var l = links[li];
    var a = nodes[l[0]], b = nodes[l[1]];
    var rel = D.relations[l[2]];
    var note = (S.relationNotes[rel] || '{a} is connected to {b}.').replace('{a}', '<b>' + esc(a.l) + '</b>').replace('{b}', '<b>' + esc(b.l) + '</b>');
    var h = '<div class="kicker">Connection · ' + esc(rel.replace(/_/g, ' ')) + '</div>';
    h += '<h2>' + esc(a.l) + ' <span class="arrow">→</span> ' + esc(b.l) + '</h2>';
    h += '<p>' + note + '</p>';
    h += '<dl class="fact"><dt>Certainty</dt><dd>' + esc(S.confidenceNotes[l[3]] || '') + (l[3] ? ' (' + Math.round(l[4] * 100) + '%)' : '') + '</dd>';
    var f = l[5] >= 0 ? D.files[l[5]] : '';
    if (f) h += '<dt>Evidence</dt><dd><span class="code">' + esc(f) + (l[6] ? ':' + esc(l[6]) : '') + '</span></dd>';
    if (l[7]) h += '<dt>Context</dt><dd>' + esc(l[7]) + '</dd>';
    h += '</dl><h3>The two ends</h3><ul class="linklist">' + nodeLine(l[0], 'from') + nodeLine(l[1], 'to') + '</ul>';
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
  }

  function showAgg(ai) {
    var g = aggLinks[ai];
    var a = nodes[g[0]], b = nodes[g[1]];
    var h = '<div class="kicker">File-level connection</div>';
    h += '<h2>' + esc(a.l) + ' <span class="arrow">↔</span> ' + esc(b.l) + '</h2>';
    h += '<p>The contents of these two files are connected <b>' + fmt(g[2]) + '</b> times. ' +
      'This line sums up all those individual connections; switch to <b>Everything</b> to see each one.</p>';
    h += '<h3>Made of</h3><dl class="fact">';
    Object.keys(g[3]).forEach(function (k) { h += '<dt>' + esc(k.replace(/_/g, ' ')) + '</dt><dd>' + fmt(g[3][k]) + '</dd>'; });
    h += '</dl><h3>The two files</h3><ul class="linklist">' + nodeLine(g[0]) + nodeLine(g[1]) + '</ul>';
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
  }

  function showCommunity(cid) {
    var c = commById[cid];
    var h = '<div class="kicker" style="color:' + c.color + '">Block · community ' + cid + '</div>';
    h += '<h2 style="color:' + c.color + '">' + esc(c.name) + '</h2>';
    h += '<p>' + esc(S.communityNotes[cid] || ('A group of ' + c.size + ' closely connected items' + (c.files.length ? ', mostly from ' + c.files[0] : '') + '.')) + '</p>';
    h += '<dl class="fact"><dt>Members</dt><dd>' + fmt(c.size) + '</dd>';
    if (c.cohesion != null) h += '<dt>Cohesion</dt><dd>' + c.cohesion + ' <span class="muted small">how tightly the members link to each other (0–1)</span><div class="bar"><i style="width:' + Math.round(clamp(c.cohesion / 0.2, 0, 1) * 100) + '%;background:' + c.color + '"></i></div></dd>';
    h += '</dl>';
    var kinds = Object.keys(c.kinds).sort(function (a, b) { return c.kinds[b] - c.kinds[a]; });
    kinds.forEach(function (k) { h += '<span class="chip plain">' + esc(S.kindLabels[k] || k) + ' · ' + c.kinds[k] + '</span>'; });
    h += '<div class="btnrow"><button class="btn" data-action="fly-comm" data-comm="' + cid + '">Fly to</button>' +
      '<button class="btn' + (isolateComm === cid ? ' on' : '') + '" data-action="isolate" data-comm="' + cid + '">' + (isolateComm === cid ? 'Show all blocks' : 'Only this block') + '</button></div>';
    h += '<h3>Main members</h3><ul class="linklist">';
    c.top.forEach(function (i) { h += nodeLine(i); });
    h += '</ul>';
    if (c.files.length) {
      h += '<h3>Files</h3>';
      c.files.forEach(function (f) { h += '<div class="small" style="margin-bottom:6px"><span class="code">' + esc(f) + '</span></div>'; });
    }
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
  }

  function showHyper(k) {
    var he = D.hyperedges[k];
    var h = '<div class="kicker alt">Group flow · from ' + esc(he.file) + '</div><h2>' + esc(he.label) + '</h2>';
    h += '<p class="muted">A set of items the documentation describes as working together as one flow.</p><ul class="linklist">';
    he.nodes.forEach(function (i) { h += nodeLine(i); });
    h += '</ul>';
    panelBody.innerHTML = h;
    panelBody.scrollTop = 0;
    sel = null; hi = { nodes: {}, links: {}, aggs: {} };
    he.nodes.forEach(function (i) { hi.nodes[i] = 2; });
    for (var li = 0; li < L; li++) if (linkVis[li] && hi.nodes[links[li][0]] && hi.nodes[links[li][1]]) hi.links[li] = 1;
    applyStyles();
    ring.visible = false;
  }

  // ------------------------------------------------------------------ selection
  function select(s, opts) {
    opts = opts || {};
    sel = s;
    if (!sel) { ring.visible = false; restyle(); showHelp(); updateLegend(); return; }
    if (sel.type === 'node') {
      if (!nodeVis[sel.idx]) {
        if (isolateComm != null && nodes[sel.idx].c !== isolateComm) { isolateComm = null; }
        if (level !== 'full' && !OVERVIEW_KINDS[nodes[sel.idx].k]) { setLevel('full', true); }
        computeVisibility();
      }
      showNode(sel.idx);
      if (opts.fly !== false) flyTo(worldOf(nodes[sel.idx]), Math.max(170, nodes[sel.idx].rad * 24), 1000);
    } else if (sel.type === 'link') {
      showLink(sel.idx);
      if (opts.fly) {
        var l = links[sel.idx];
        var mid = worldOf(nodes[l[0]]).add(worldOf(nodes[l[1]], _v2)).multiplyScalar(0.5);
        flyTo(mid, Math.max(120, linkG[sel.idx].len * 1.6), 1000);
      }
    } else if (sel.type === 'agg') {
      showAgg(sel.idx);
    } else if (sel.type === 'community') {
      showCommunity(sel.idx);
      if (opts.fly !== false) {
        var c = commById[sel.idx];
        flyTo(graphGroup.localToWorld(new THREE.Vector3(c.cx, c.cy, c.cz)), c.r * 3.2 + 120, 1100);
      }
    }
    restyle();
    updateLegend();
  }

  function updateRing(now) {
    if (!sel || sel.type !== 'node' || !nodeVis[sel.idx]) { ring.visible = false; return; }
    var n = nodes[sel.idx];
    ring.visible = true;
    worldOf(n, ring.position);
    var s = n.rad * graphGroup.scale.x * (1.15 + 0.12 * Math.sin(now / 260));
    ring.scale.set(s, s, s);
    ring.material.color.copy(n.col).lerp(new THREE.Color(0xffffff), 0.5);
    ring.quaternion.copy(camera.quaternion);
  }

  // ------------------------------------------------------------------ legend / status / search
  var legendExpanded = false;
  function updateLegend() {
    var el = $('legend');
    var ranked = comms.slice().sort(function (a, b) { return b.size - a.size; });
    var list = legendExpanded ? ranked : ranked.slice(0, 12);
    var h = '<span class="kicker">Blocks · ' + comms.length + '</span>';
    list.forEach(function (c) {
      var on = (sel && sel.type === 'community' && sel.idx === c.id) || isolateComm === c.id;
      h += '<div class="row' + (on ? ' on' : '') + '" data-comm="' + c.id + '"><span class="sw" style="color:' + c.color + '"></span><span>' + esc(c.name) + '</span><span class="n">' + c.size + '</span></div>';
    });
    h += '<div class="more" data-action="legend-toggle">' + (legendExpanded ? 'Show fewer' : 'Show all ' + comms.length + ' blocks') + '</div>';
    el.innerHTML = h;
  }
  function updateStatus() {
    $('status').textContent = fmt(visibleCount) + ' of ' + fmt(N) + ' items · ' + fmt(visibleLinkCount) + ' connections shown · ' +
      comms.length + ' blocks · built from commit ' + (D.meta.commit || '').slice(0, 7) + ' · ' + (D.meta.generated || '').slice(0, 10);
  }

  var searchInput = $('search'), searchList = $('searchList');
  var searchActive = -1, searchItems = [];
  function runSearch() {
    var q = searchInput.value.trim().toLowerCase();
    searchItems = [];
    searchActive = -1;
    if (q.length < 2) { searchList.hidden = true; return; }
    comms.forEach(function (c) { if (c.name.toLowerCase().indexOf(q) >= 0) searchItems.push({ type: 'community', idx: c.id, text: c.name, sub: c.size + ' members', score: 1000 + c.size }); });
    for (var i = 0; i < N && searchItems.length < 400; i++) {
      var n = nodes[i];
      var l = n.l.toLowerCase();
      var p = l.indexOf(q);
      if (p < 0) continue;
      searchItems.push({ type: 'node', idx: i, text: n.l, sub: n.file || S.kindLabels[n.k], score: (p === 0 ? 200 : 0) + n.d + (n.k === 'file' ? 50 : 0) });
    }
    searchItems.sort(function (a, b) { return b.score - a.score; });
    searchItems = searchItems.slice(0, 14);
    if (!searchItems.length) { searchList.innerHTML = '<li class="muted">No match</li>'; searchList.hidden = false; return; }
    searchList.innerHTML = searchItems.map(function (it, k) {
      var kind = it.type === 'community' ? 'Block' : (S.kindLabels[nodes[it.idx].k] || nodes[it.idx].k);
      return '<li data-k="' + k + '"><span class="k">' + esc(kind) + '</span><span>' + esc(it.text) + '</span><span class="sub">' + esc(it.sub) + '</span></li>';
    }).join('');
    searchList.hidden = false;
  }
  function chooseSearch(k) {
    var it = searchItems[k];
    if (!it) return;
    searchList.hidden = true;
    searchInput.value = it.text;
    searchInput.blur();
    select({ type: it.type, idx: it.idx });
  }
  searchInput.addEventListener('input', runSearch);
  searchInput.addEventListener('focus', runSearch);
  searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!searchItems.length) return;
      searchActive = (searchActive + (e.key === 'ArrowDown' ? 1 : -1) + searchItems.length) % searchItems.length;
      Array.prototype.forEach.call(searchList.children, function (li, k) { li.classList.toggle('active', k === searchActive); });
    } else if (e.key === 'Enter') { chooseSearch(searchActive >= 0 ? searchActive : 0); }
    else if (e.key === 'Escape') { searchList.hidden = true; searchInput.blur(); }
    e.stopPropagation();
  });
  searchList.addEventListener('mousedown', function (e) {
    var li = e.target.closest('li[data-k]');
    if (li) { e.preventDefault(); chooseSearch(+li.dataset.k); }
  });
  document.addEventListener('mousedown', function (e) { if (!e.target.closest('.search')) searchList.hidden = true; });

  // ------------------------------------------------------------------ UI wiring
  function setLevel(lv, silent) {
    level = lv;
    Array.prototype.forEach.call($('detailSeg').children, function (b) { b.classList.toggle('on', b.dataset.level === lv); });
    if (!silent) { computeVisibility(); restyle(); if (sel && sel.type === 'node') showNode(sel.idx); }
    if (!silent) toast(lv === 'full' ? 'Everything: all ' + fmt(N) + ' items and ' + fmt(L) + ' connections' : 'Overview: files and documentation only');
  }
  $('detailSeg').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-level]');
    if (b && b.dataset.level !== level) setLevel(b.dataset.level);
  });
  $('labelsBtn').addEventListener('click', function () { labelsOn = !labelsOn; $('labelsBtn').classList.toggle('on', labelsOn); });
  function setSpin(on) { controls.autoRotate = on; $('spinBtn').classList.toggle('on', on); }
  $('spinBtn').addEventListener('click', function () { setSpin(!controls.autoRotate); });
  $('resetBtn').addEventListener('click', function () { isolateComm = null; computeVisibility(); select(null); fitVisible(1200); });
  $('replayBtn').addEventListener('click', function () { startIntro(); });
  $('helpBtn').addEventListener('click', function () { select(null); });
  $('skipBtn').addEventListener('click', function () { enterExplore(true); });
  (function () {
    var b = $('backBtn'); if (b) b.addEventListener('click', prevStep);
    var p = $('pauseBtn'); if (p) p.addEventListener('click', togglePause);
  })();

  function onPanelClick(e) {
    var t = e.target.closest('[data-action],[data-node],[data-link],[data-agg],[data-comm],[data-hyper]');
    if (!t) return;
    var act = t.dataset.action;
    if (act === 'fly') { var n = nodes[+t.dataset.node]; flyTo(worldOf(n), Math.max(170, n.rad * 24), 1000); return; }
    if (act === 'fly-comm') { var c = commById[+t.dataset.comm]; flyTo(graphGroup.localToWorld(new THREE.Vector3(c.cx, c.cy, c.cz)), c.r * 3.2 + 120, 1100); return; }
    if (act === 'isolate') {
      var cid = +t.dataset.comm;
      isolateComm = isolateComm === cid ? null : cid;
      computeVisibility();
      select({ type: 'community', idx: cid });
      if (isolateComm != null) toast('Showing only "' + commById[cid].name + '". Reset view brings everything back.');
      return;
    }
    if (act === 'legend-toggle') { legendExpanded = !legendExpanded; updateLegend(); return; }
    if (t.dataset.hyper != null) { showHyper(+t.dataset.hyper); return; }
    if (t.dataset.node != null) { select({ type: 'node', idx: +t.dataset.node }); return; }
    if (t.dataset.link != null) { select({ type: 'link', idx: +t.dataset.link, }, { fly: true }); return; }
    if (t.dataset.agg != null) { select({ type: 'agg', idx: +t.dataset.agg }); return; }
    if (t.dataset.comm != null) { select({ type: 'community', idx: +t.dataset.comm }); return; }
  }
  panelBody.addEventListener('click', onPanelClick);
  $('legend').addEventListener('click', onPanelClick);

  // Pointer interaction on the canvas.
  var pointer = { x: 0, y: 0, downX: 0, downY: 0, down: false, moved: false, need: false, lastClick: 0 };
  canvas.addEventListener('pointermove', function (e) {
    pointer.x = e.clientX; pointer.y = e.clientY; pointer.need = true;
    if (pointer.down && (Math.abs(e.clientX - pointer.downX) > 4 || Math.abs(e.clientY - pointer.downY) > 4)) { pointer.moved = true; canvas.classList.add('dragging'); }
  });
  canvas.addEventListener('pointerdown', function (e) {
    pointer.down = true; pointer.moved = false; pointer.downX = e.clientX; pointer.downY = e.clientY;
    if (mode === 'explore' && controls.autoRotate) setSpin(false);
  });
  canvas.addEventListener('pointerup', function (e) {
    canvas.classList.remove('dragging');
    var wasDown = pointer.down;
    pointer.down = false;
    if (!wasDown || pointer.moved || mode !== 'explore' || e.button !== 0) return;
    var now = performance.now();
    var dbl = now - pointer.lastClick < 380;
    pointer.lastClick = now;
    var ni = pickNode(e.clientX, e.clientY);
    if (ni >= 0) {
      if (dbl) { flyTo(worldOf(nodes[ni]), Math.max(110, nodes[ni].rad * 14), 900); return; }
      select({ type: 'node', idx: ni }, { fly: false });
      return;
    }
    var lk = pickLink(e.clientX, e.clientY);
    if (lk) { select(lk, { fly: false }); return; }
    if (!dbl) select(null);
  });
  canvas.addEventListener('pointerleave', function () { hoverNode = -1; hoverLink = -1; hoverAgg = -1; $('tooltip').hidden = true; });
  canvas.addEventListener('contextmenu', function (e) { e.preventDefault(); });

  function updateHover() {
    if (!pointer.need || mode !== 'explore' || pointer.down) return;
    pointer.need = false;
    var tip = $('tooltip');
    var ni = pickNode(pointer.x, pointer.y);
    var lk = ni < 0 ? pickLink(pointer.x, pointer.y) : null;
    var nh = ni, nl = lk && lk.type === 'link' ? lk.idx : -1, na = lk && lk.type === 'agg' ? lk.idx : -1;
    if (nh !== hoverNode || nl !== hoverLink || na !== hoverAgg) {
      hoverNode = nh; hoverLink = nl; hoverAgg = na;
      applyStyles();
    }
    canvas.classList.toggle('pick', ni >= 0 || !!lk);
    if (ni >= 0) {
      var n = nodes[ni];
      tip.innerHTML = '<div class="tt-kicker">' + esc(S.kindLabels[n.k] || n.k) + ' · ' + esc(commById[n.c].name) + '</div><div>' + esc(n.l) + '</div>' +
        (n.file ? '<div class="tt-sub">' + esc(n.file) + (n.loc ? ':' + esc(n.loc) : '') + '</div>' : '');
    } else if (lk && lk.type === 'link') {
      var l = links[lk.idx];
      tip.innerHTML = '<div class="tt-kicker">connection · ' + esc(D.relations[l[2]].replace(/_/g, ' ')) + '</div><div>' + esc(nodes[l[0]].l) + ' <span class="arrow">→</span> ' + esc(nodes[l[1]].l) + '</div>';
    } else if (lk && lk.type === 'agg') {
      var g = aggLinks[lk.idx];
      tip.innerHTML = '<div class="tt-kicker">file-level connection · ' + g[2] + ' links</div><div>' + esc(nodes[g[0]].l) + ' <span class="arrow">↔</span> ' + esc(nodes[g[1]].l) + '</div>';
    } else { tip.hidden = true; return; }
    tip.hidden = false;
    var W = window.innerWidth, H = window.innerHeight;
    var x = pointer.x + 16, y = pointer.y + 18;
    if (x + tip.offsetWidth > W - 460) x = pointer.x - tip.offsetWidth - 14;
    if (y + tip.offsetHeight > H - 10) y = pointer.y - tip.offsetHeight - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }

  window.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (mode === 'intro') {
      if (e.key === 'Escape') { e.preventDefault(); enterExplore(true); }
      else if (e.key === ' ' || e.key === 'Enter' || e.key === 'ArrowRight') { e.preventDefault(); nextStep(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); prevStep(); }
      else if (e.key === 'p' || e.key === 'P') { e.preventDefault(); togglePause(); }
      return;
    }
    if (mode !== 'explore') return;
    if (e.key === 'Escape') { if (isolateComm != null) { isolateComm = null; computeVisibility(); } select(null); }
    else if (e.key === 'f' || e.key === 'F') fitVisible(1000);
    else if (e.key === 'r' || e.key === 'R') setSpin(!controls.autoRotate);
    else if (e.key === 'l' || e.key === 'L') $('labelsBtn').click();
  });

  function resize() {
    var w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    haloMat.uniforms.uScale.value = (h * renderer.getPixelRatio()) / (2 * Math.tan(camera.fov * Math.PI / 360));
  }
  window.addEventListener('resize', resize);
  resize();

  // ------------------------------------------------------------------ intro
  var intro = { group: new THREE.Group(), blocks: {}, links: [], step: -1, timer: null, sparks: [] };
  scene.add(intro.group);
  var glowTex = (function () {
    var c = document.createElement('canvas'); c.width = c.height = 128;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grd.addColorStop(0.6, 'rgba(255,255,255,0.12)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
    var t = new THREE.CanvasTexture(c);
    return t;
  })();
  var blockGeo = new THREE.SphereGeometry(15, 32, 24);
  var torusGeo = new THREE.TorusGeometry(24, 0.55, 8, 72);
  var DORMANT = { scale: 1.0, pos: new THREE.Vector3(0, 70, -640), opacity: { node: 0.42, halo: 0.5, link: 0.15 } };

  function clearIntroObjects() {
    while (intro.group.children.length) {
      var o = intro.group.children[0];
      intro.group.remove(o);
    }
    intro.blocks = {}; intro.links = []; intro.sparks = []; introLabels = [];
  }
  function buildIntroObjects() {
    clearIntroObjects();
    blocks.forEach(function (b) {
      var holder = new THREE.Group();
      holder.position.set(b.pos[0], b.pos[1], b.pos[2]);
      holder.scale.setScalar(0.0001);
      holder.visible = false;
      var core = new THREE.Mesh(blockGeo, new THREE.MeshLambertMaterial({ color: b.col, emissive: b.col, emissiveIntensity: 0.45 }));
      var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: b.col, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
      glow.scale.setScalar(110);
      var ringA = new THREE.Mesh(torusGeo, new THREE.MeshBasicMaterial({ color: b.col, transparent: true, opacity: 0.7 }));
      var ringB = ringA.clone();
      ringA.rotation.x = Math.PI / 2.3; ringB.rotation.y = Math.PI / 2.6; ringB.rotation.x = 0.4;
      holder.add(core); holder.add(glow); holder.add(ringA); holder.add(ringB);
      intro.group.add(holder);
      intro.blocks[b.key] = { holder: holder, core: core, glow: glow, rings: [ringA, ringB], block: b };
      introLabels.push({ obj: holder, text: b.name, color: b.color });
    });
  }
  function spawnBlock(key) {
    var ib = intro.blocks[key];
    if (!ib || ib.holder.visible) return;
    ib.holder.visible = true;
    tween(950, function (k) { ib.holder.scale.setScalar(Math.max(0.0001, k)); }, { ease: ease.outBack, tag: 'intro' });
  }
  function drawIntroLink(fromKey, toKey, delay) {
    var a = intro.blocks[fromKey], b = intro.blocks[toKey];
    if (!a || !b) return;
    spawnBlock(fromKey);
    var pa = a.holder.position, pb = b.holder.position;
    var dir = pb.clone().sub(pa);
    var len = dir.length();
    var mesh = new THREE.Mesh(cylGeo, new THREE.MeshBasicMaterial({ color: b.block.col.clone().lerp(a.block.col, 0.5).lerp(new THREE.Color(0xffffff), 0.35), transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
    mesh.position.copy(pa);
    mesh.quaternion.setFromUnitVectors(Y_AXIS, dir.clone().normalize());
    mesh.scale.set(1.6, 0.001, 1.6);
    intro.group.add(mesh);
    var spark = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: 0xffffff, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false }));
    spark.scale.setScalar(26);
    spark.visible = false;
    intro.group.add(spark);
    var rec = { mesh: mesh, spark: spark, a: pa.clone(), b: pb.clone(), len: len, dir: dir.clone().normalize(), phase: Math.random(), live: false };
    intro.links.push(rec);
    tween(900, function (k) { mesh.scale.y = Math.max(0.001, len * k); }, { ease: ease.outCubic, delay: delay || 0, tag: 'intro', done: function () { rec.live = true; spark.visible = true; } });
  }
  function updateIntroSparks(now) {
    intro.links.forEach(function (r) {
      if (!r.live) return;
      var t = ((now / 1700) + r.phase) % 1;
      r.spark.position.copy(r.a).addScaledVector(r.dir, r.len * t);
      r.spark.material.opacity = 0.35 + 0.65 * Math.sin(t * Math.PI);
    });
    Object.keys(intro.blocks).forEach(function (k) {
      var ib = intro.blocks[k];
      if (!ib.holder.visible) return;
      ib.rings[0].rotation.z += 0.012; ib.rings[1].rotation.z -= 0.009;
      ib.glow.material.opacity = 0.7 + 0.2 * Math.sin(now / 500 + ib.holder.position.x);
    });
  }

  var introCam = { target: new THREE.Vector3(0, 40, 0), pos: new THREE.Vector3(0, 90, 880), wantTarget: new THREE.Vector3(0, 40, 0), wantDist: 880 };
  function introCameraUpdate(now) {
    introCam.target.lerp(introCam.wantTarget, 0.035);
    var drift = new THREE.Vector3(Math.sin(now / 5200) * 70, 40 + Math.cos(now / 6100) * 30, 0);
    var want = introCam.target.clone().add(drift).setZ(introCam.wantDist);
    camera.position.lerp(want, 0.035);
    camera.lookAt(introCam.target);
    controls.target.copy(introCam.target);
  }
  function setDormant(k) {
    // k = 1 → dormant background nebula, k = 0 → live map.
    graphGroup.scale.setScalar(lerp(1, DORMANT.scale, k));
    graphGroup.position.lerpVectors(new THREE.Vector3(0, 0, 0), DORMANT.pos, k);
    nodeMat.opacity = lerp(1, DORMANT.opacity.node, k);
    haloMat.uniforms.uGlobal.value = lerp(1, DORMANT.opacity.halo, k);
    linkMat.opacity = lerp(0.6, DORMANT.opacity.link, k);
  }

  var progressEl = $('progress');
  function buildProgress() {
    progressEl.innerHTML = S.steps.map(function (st, i) {
      var name = (st.title || ('Step ' + (i + 1))).replace(/"/g, '&quot;');
      return '<span tabindex="0" role="button" title="' + name + '"></span>';
    }).join('');
    Array.prototype.forEach.call(progressEl.children, function (sp, i) {
      sp.addEventListener('click', function () { goToStep(i); });
      sp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToStep(i); }
      });
    });
  }
  function setCaption(st) {
    var cap = $('caption');
    cap.classList.remove('show');
    setTimeout(function () {
      $('capKicker').textContent = st.kicker || '';
      $('capTitle').textContent = st.title || '';
      $('capText').textContent = st.text || '';
      void cap.offsetWidth;
      cap.classList.add('show');
    }, 220);
  }
  function runStep(st, idx) {
    setCaption(st);
    Array.prototype.forEach.call(progressEl.children, function (sp, k) {
      sp.className = k < idx ? 'done' : '';
      if (k === idx) {
        sp.style.transitionDuration = '0s';
        sp.className = '';
        void sp.offsetWidth;
        sp.style.transitionDuration = st.duration + 'ms';
        sp.className = 'active';
      }
    });
    if (st.kind === 'block') {
      var b = blockByKey[st.block];
      spawnBlock(st.block);
      (st.from || []).forEach(function (f, k) { drawIntroLink(f, st.block, 150 + k * 250); });
      introCam.wantTarget.set(b.pos[0] * 0.28, 40 + b.pos[1] * 0.28, 0);
      introCam.wantDist = 820;
    } else if (st.kind === 'link') {
      (st.links || []).forEach(function (pair, k) { drawIntroLink(pair[0], pair[1], 100 + k * 500); });
      introCam.wantTarget.set(0, 40, 0);
      introCam.wantDist = 880;
    } else if (st.kind === 'outro') {
      blocks.forEach(function (b) { spawnBlock(b.key); });
      introCam.wantTarget.set(0, 40, 0);
      introCam.wantDist = 940;
      var g0 = { s: graphGroup.scale.x };
      tween(st.duration - 300, function (k) { graphGroup.scale.setScalar(lerp(g0.s, 1.05, k)); }, { tag: 'intro', ease: ease.inOutCubic });
    } else {
      introCam.wantTarget.set(0, 40, 0);
      introCam.wantDist = 880;
    }
  }
  function nextStep() {
    if (mode !== 'intro') return;
    clearTimeout(intro.timer);
    intro.step++;
    if (intro.step >= S.steps.length) { enterExplore(false); return; }
    goToStep(intro.step);
  }

  // Jump to an arbitrary step. Blocks already spawned stay spawned - the tour
  // is additive by design, so stepping back re-tells the story without
  // tearing the map down.
  function goToStep(i) {
    if (mode !== 'intro') return;
    clearTimeout(intro.timer);
    intro.step = Math.max(0, Math.min(i, S.steps.length - 1));
    var st = S.steps[intro.step];
    runStep(st, intro.step);
    intro.stepStart = performance.now();
    intro.stepDur = st.duration;
    intro.elapsed = 0;
    if (intro.paused) { freezeBar(); return; }
    intro.timer = setTimeout(nextStep, st.duration);
  }

  function prevStep() {
    if (mode !== 'intro') return;
    goToStep(intro.step - 1);
  }

  function activeBar() { return progressEl.children[intro.step] || null; }

  function freezeBar() {
    var sp = activeBar();
    if (!sp) return;
    var w = '0px';
    try { w = window.getComputedStyle(sp, '::after').width || '0px'; } catch (e) {}
    sp.style.setProperty('--frozen', w);
    sp.classList.add('paused');
  }

  function togglePause() {
    if (mode !== 'intro') return;
    var btn = $('pauseBtn');
    if (!intro.paused) {
      intro.paused = true;
      clearTimeout(intro.timer);
      intro.elapsed = performance.now() - (intro.stepStart || performance.now());
      freezeBar();
      if (btn) { btn.textContent = 'Resume'; btn.classList.add('on'); }
    } else {
      intro.paused = false;
      var rem = Math.max(120, (intro.stepDur || 0) - (intro.elapsed || 0));
      var sp = activeBar();
      if (sp) {
        sp.classList.remove('paused');
        sp.style.transitionDuration = rem + 'ms';
      }
      intro.stepStart = performance.now() - (intro.elapsed || 0);
      intro.timer = setTimeout(nextStep, rem);
      if (btn) { btn.textContent = 'Pause'; btn.classList.remove('on'); }
    }
  }

  function startIntro() {
    mode = 'intro';
    clearTimeout(intro.timer);
    intro.paused = false;
    intro.elapsed = 0;
    (function () {
      var pb = $('pauseBtn');
      if (pb) { pb.textContent = 'Pause'; pb.classList.remove('on'); }
    })();
    killTweens();
    sel = null; hi = null; isolateComm = null; hoverNode = hoverLink = hoverAgg = -1;
    ring.visible = false;
    $('tooltip').hidden = true;
    setLevel('full', true);
    computeVisibility();
    applyStyles();
    controls.enabled = false;
    setSpin(false);
    setDormant(1);
    graphGroup.rotation.set(0.15, 0, 0);
    buildIntroObjects();
    buildProgress();
    camera.position.copy(introCam.pos);
    introCam.target.set(0, 40, 0);
    introCam.wantTarget.set(0, 40, 0);
    introCam.wantDist = 880;
    camera.lookAt(introCam.target);
    var ex = $('explore');
    ex.classList.remove('show');
    ex.hidden = true;
    var inEl = $('intro');
    inEl.hidden = false;
    void inEl.offsetWidth;
    inEl.classList.remove('fade');
    $('caption').classList.remove('show');
    intro.step = -1;
    intro.timer = setTimeout(nextStep, 500);
  }

  function enterExplore(fast) {
    if (mode === 'explore') return;
    var wasIntro = mode === 'intro';
    mode = 'transition';
    clearTimeout(intro.timer);
    killTweens('intro');
    killTweens('cam');
    var dur = fast ? 1500 : 2800;
    var k0 = { s: graphGroup.scale.x, p: graphGroup.position.clone(), rx: graphGroup.rotation.x, ry: graphGroup.rotation.y,
      n: nodeMat.opacity, h: haloMat.uniforms.uGlobal.value, l: linkMat.opacity };
    var camP0 = camera.position.clone(), camT0 = controls.target.clone();
    var fit = computeFit(false) || { target: new THREE.Vector3(0, 0, 0), pos: new THREE.Vector3(0.18, 0.22, 1).normalize().multiplyScalar(1350) };
    var camT1 = fit.target, camP1 = fit.pos;
    $('intro').classList.add('fade');
    $('caption').classList.remove('show');
    tween(dur, function (k) {
      graphGroup.scale.setScalar(lerp(k0.s, 1, k));
      graphGroup.position.lerpVectors(k0.p, new THREE.Vector3(0, 0, 0), k);
      graphGroup.rotation.x = lerp(k0.rx, 0, k);
      graphGroup.rotation.y = lerp(k0.ry, 0, k);
      nodeMat.opacity = lerp(k0.n, 1, k);
      haloMat.uniforms.uGlobal.value = lerp(k0.h, 1, k);
      linkMat.opacity = lerp(k0.l, 0.6, k);
      camera.position.lerpVectors(camP0, camP1, k);
      controls.target.lerpVectors(camT0, camT1, k);
      camera.lookAt(controls.target);
      graphGroup.updateMatrixWorld(true);
      Object.keys(intro.blocks).forEach(function (key) {
        var ib = intro.blocks[key];
        if (!ib.holder.visible) return;
        if (!ib.start) ib.start = ib.holder.position.clone();
        if (ib.block.anchorIdx >= 0) {
          worldOf(nodes[ib.block.anchorIdx], _v2);
          ib.holder.position.lerpVectors(ib.start, _v2, ease.inOutCubic(k));
        }
        ib.holder.scale.setScalar(Math.max(0.0001, 1 - Math.pow(k, 1.6)));
      });
      intro.links.forEach(function (r) { r.mesh.material.opacity = 0.85 * (1 - Math.min(1, k * 2.2)); r.spark.visible = k < 0.4; });
    }, { ease: ease.inOutCubic, tag: 'transition', done: function () {
      clearIntroObjects();
      $('intro').hidden = true;
      mode = 'explore';
      controls.enabled = true;
      controls.update();
      setSpin(true);
      if (!wasIntro) return;
    } });
    setTimeout(function () {
      var ex = $('explore');
      ex.hidden = false;
      void ex.offsetWidth;
      ex.classList.add('show');
    }, fast ? 200 : 900);
    computeVisibility();
    select(null);
  }

  // ------------------------------------------------------------------ main loop
  var last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    var dt = Math.min(64, now - last);
    last = now;
    stepTweens(now);
    if (mode === 'intro') {
      graphGroup.rotation.y += dt * 0.00007;
      introCameraUpdate(now);
      updateIntroSparks(now);
    } else if (mode === 'transition') {
      updateIntroSparks(now);
    } else {
      controls.update();
      updateHover();
    }
    updateRing(now);
    updateSparks(now);
    renderer.render(scene, camera);
    updateLabels();
  }

  // Small scripting surface for tests and power users (browser console).
  window.ZWMAP = {
    idx: function (id) { return idIndex[id] != null ? idIndex[id] : -1; },
    select: function (type, idx, opts) { select({ type: type, idx: idx }, opts); },
    level: setLevel,
    fit: function () { fitVisible(800); },
    intro: startIntro,
    explore: function () { enterExplore(true); },
    next: nextStep,
    mode: function () { return mode; },
    stats: function () { return { nodes: N, links: L, visible: visibleCount, visibleLinks: visibleLinkCount, step: intro.step }; }
  };

  // Boot: build the map once, then start the tour.
  computeVisibility();
  applyStyles();
  updateLegend();
  showHelp();
  startIntro();
  requestAnimationFrame(frame);
})();
