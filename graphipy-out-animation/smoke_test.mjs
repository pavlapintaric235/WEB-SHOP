// Headless-Chrome smoke test for the living map.
//   node smoke_test.mjs            (set CHROME=<path> if Chrome is not in the default place)
// Loads index.html over file://, drives the tour and the explorer through the ZWMAP
// helpers, records console errors and exceptions, and writes screenshots to smoke_shots/.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import http from 'node:http';

const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME = process.env.CHROME || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'
].find(existsSync);
if (!CHROME) { console.error('No Chrome/Edge found. Set CHROME=<path to browser executable>.'); process.exit(2); }
const URL = pathToFileURL(join(HERE, 'index.html')).href;
const OUT = join(HERE, 'smoke_shots');
const PORT = 9333 + Math.floor(Math.random() * 500);
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--window-size=1600,900', '--hide-scrollbars',
  '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
  '--no-first-run', '--no-default-browser-check', '--user-data-dir=' + join(OUT, 'profile'), 'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
function getJSON(path) {
  return new Promise((res, rej) => http.get({ host: '127.0.0.1', port: PORT, path }, r => {
    let b = ''; r.on('data', d => b += d); r.on('end', () => { try { res(JSON.parse(b)); } catch (e) { rej(e); } });
  }).on('error', rej));
}
let targets = null;
for (let i = 0; i < 60; i++) { try { targets = await getJSON('/json'); break; } catch { await sleep(250); } }
if (!targets) { console.error('Chrome did not start'); chrome.kill(); process.exit(1); }
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
let seq = 0; const pending = new Map(); const logs = []; let failures = 0;
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.consoleAPICalled') logs.push(`[console.${m.params.type}] ` + m.params.args.map(a => a.value ?? a.description ?? JSON.stringify(a)).join(' '));
  if (m.method === 'Runtime.exceptionThrown') { failures++; logs.push('[EXCEPTION] ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text)); }
  if (m.method === 'Log.entryAdded') { if (m.params.entry.level === 'error') failures++; logs.push(`[log.${m.params.entry.level}] ${m.params.entry.text} ${m.params.entry.url || ''}`); }
};
const send = (method, params = {}) => new Promise(res => { const id = ++seq; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
async function evalJS(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) { failures++; return 'EVAL ERROR: ' + (r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text); }
  return r.result?.result?.value;
}
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(OUT, name + '.png'), Buffer.from(r.result.data, 'base64'));
  console.log('screenshot', name);
}
function expect(label, ok) { console.log((ok ? 'ok   ' : 'FAIL ') + label); if (!ok) failures++; }

await send('Runtime.enable'); await send('Log.enable'); await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: URL });
await sleep(2500);
expect('tour starts in intro mode', await evalJS('window.ZWMAP && ZWMAP.mode()') === 'intro');
const stats = await evalJS('ZWMAP.stats()');
expect('all nodes visible during the tour', stats && stats.visible === stats.nodes && stats.nodes > 0);
await shot('01-intro-title');
await evalJS('ZWMAP.next(); ZWMAP.next(); ZWMAP.next();');
await sleep(2200);
await shot('02-intro-blocks');
await evalJS('for (let i = 0; i < 6; i++) ZWMAP.next();');
await sleep(2200);
await shot('03-intro-outro');
await evalJS('ZWMAP.explore()');
await sleep(2500);
expect('explorer reached after skip', await evalJS('ZWMAP.mode()') === 'explore');
expect('fonts loaded', (await evalJS('[...document.fonts].filter(f => f.status === "loaded").map(f => f.family)')).some(f => /Orbitron|Exo/.test(f)));
await shot('04-explore');
await evalJS('ZWMAP.select("node", ZWMAP.idx("db"))');
await sleep(1500);
expect('node panel shows the file', (await evalJS('document.querySelector("#panelBody h2")?.textContent') || '').includes("db.py"));
await shot('05-node');
await evalJS('ZWMAP.select("community", 1)');
await sleep(1200);
expect('community panel shows a block', (await evalJS('document.querySelector("#panelBody .kicker")?.textContent') || '').toLowerCase().includes('block'));
await evalJS('ZWMAP.level("overview")');
await sleep(800);
const ov = await evalJS('ZWMAP.stats()');
expect('overview hides most nodes', ov && ov.visible < ov.nodes / 3);
await shot('06-overview');
await evalJS('ZWMAP.level("full")');
await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 600, y: 450 });
await sleep(500);
await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: 600, y: 450, button: 'left', clickCount: 1 });
await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: 600, y: 450, button: 'left', clickCount: 1 });
await sleep(800);
console.log('panel after a real click at the centre:', await evalJS('document.querySelector("#panelBody .kicker")?.textContent'));
await shot('07-after-click');
console.log('--- browser console ---');
logs.forEach(l => console.log(l));
console.log('--- end ---');
console.log(failures ? `FAILED (${failures})` : 'PASSED');
ws.close(); chrome.kill();
process.exit(failures ? 1 : 0);
