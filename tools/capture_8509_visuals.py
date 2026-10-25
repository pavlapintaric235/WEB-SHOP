"""Capture the actual local app with an isolated Chromium CDP target.

Uses the already installed browser and websockets package, no app dependency.
Employee capture performs a real worker request on the supplied scene.
"""
import base64
import json
from pathlib import Path
import time
import sys
from urllib.request import urlopen

from websockets.sync.client import connect

ROOT=Path(__file__).resolve().parents[1]


class Browser:
    def __init__(self):
        pages=json.load(urlopen('http://127.0.0.1:9230/json'))
        page=next(p for p in pages if p.get('url','').startswith('http://localhost:8509'))
        self.ws=connect(page['webSocketDebuggerUrl'],max_size=25*1024*1024);self.identity=0
    def call(self,method,params=None):
        self.identity+=1;identity=self.identity
        self.ws.send(json.dumps(dict(id=identity,method=method,params=params or {})))
        while True:
            response=json.loads(self.ws.recv())
            if response.get('id')==identity:
                if 'error' in response:raise RuntimeError(response['error'])
                return response.get('result',{})
    def js(self,expression):
        result=self.call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
        return result.get('result',{}).get('value')
    def wait(self,text,timeout=120):
        start=time.monotonic()
        while time.monotonic()-start<timeout:
            if text in (self.js('document.body.innerText') or ''):return
            time.sleep(.5)
        raise RuntimeError('UI did not show '+text+': '+str(self.js('document.body.innerText'))[:1600])
    def click(self,text):
        self.idle()
        self.js("(()=>{let e=[...document.querySelectorAll('button')].find(e=>e.innerText.trim()==="+json.dumps(text)+");if(!e)throw Error('button not found');e.click()})()")
        time.sleep(.7)
    def radio(self,text):
        self.js("(()=>{let e=[...document.querySelectorAll('label')].find(e=>e.innerText.trim()==="+json.dumps(text)+");if(!e)throw Error('radio not found');e.click()})()")
        time.sleep(1)
    def fill(self,label,text):
        self.idle()
        start=time.monotonic()
        while not self.js("[...document.querySelectorAll('input')].some(e=>e.getAttribute('aria-label')==="+json.dumps(label)+")"):
            if time.monotonic()-start>60:raise RuntimeError('Input did not render: '+label)
            time.sleep(.5)
        self.js("(()=>{let e=[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')==="+json.dumps(label)+");if(!e)throw Error('input not found');e.focus()})()")
        self.call('Input.dispatchKeyEvent',dict(type='rawKeyDown',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=2))
        self.call('Input.dispatchKeyEvent',dict(type='keyUp',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=0))
        self.call('Input.insertText',dict(text=text))
        self.js('document.activeElement.blur()')
        self.call('Input.dispatchKeyEvent',dict(type='keyDown',key='Tab',code='Tab',windowsVirtualKeyCode=9))
        self.call('Input.dispatchKeyEvent',dict(type='keyUp',key='Tab',code='Tab',windowsVirtualKeyCode=9))
        if label not in ('Username','Password'):
            self.call('Input.dispatchKeyEvent',dict(type='keyDown',key='Enter',code='Enter',windowsVirtualKeyCode=13))
            self.call('Input.dispatchKeyEvent',dict(type='keyUp',key='Enter',code='Enter',windowsVirtualKeyCode=13))
        time.sleep(.7)
        self.idle()
    def idle(self,timeout=180):
        start=time.monotonic()
        while time.monotonic()-start<timeout:
            if not self.js("document.body.innerText.split('\\n').some(t=>t.trim()==='Stop')"):return
            time.sleep(.5)
        raise RuntimeError('App did not finish its current operation')
    def screenshot(self,name):
        self.idle()
        self.js("window.getSelection()?.removeAllRanges();[...document.querySelectorAll('button')].find(e=>e.innerText.trim()===\"Don't show again\")?.click()")
        time.sleep(1)
        data=self.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']
        (ROOT/'docs'/name).write_bytes(base64.b64decode(data))
    def upload(self,path):
        start=time.monotonic();node=0
        while not node:
            document=self.call('DOM.getDocument',dict(depth=0))['root']['nodeId']
            node=self.call('DOM.querySelector',dict(nodeId=document,selector='input[type=file][accept*=".jpg"]'))['nodeId']
            if not node:
                if time.monotonic()-start>60:raise RuntimeError('Image file input not found')
                time.sleep(.5)
        self.call('DOM.setFileInputFiles',dict(nodeId=node,files=[str(path)]))
    def login(self,name):
        self.wait('Username');self.fill('Username',name);self.fill('Password',name+'123');self.click('Sign in')


def main():
    b=Browser();b.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
    if '--inspect' in sys.argv:
        print(json.dumps(dict(inputs=b.js("[...document.querySelectorAll('input')].map(e=>({aria:e.getAttribute('aria-label'),id:e.id,type:e.type}))"),text=b.js('document.body.innerText')[:1400]),ensure_ascii=True));b.ws.close();return
    if '--review-only' in sys.argv:
        b.idle();b.wait('Products in this photo')
        b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Products in this photo')).scrollIntoView({block:'start'})")
        b.screenshot('webshop_8509_employee.png');b.ws.close();return
    if '--finish-existing' in sys.argv:
        b.idle();b.wait('Products in this photo')
        b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Products in this photo'))?.scrollIntoView({block:'start'})")
        b.screenshot('webshop_8509_employee.png')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
        b.screenshot('webshop_8509_employee_mobile.png')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
    if 'Sign out' in b.js('document.body.innerText'):b.click('Sign out')
    b.login('owner');b.wait('Your products. Ready for the shelf.');b.idle()
    b.fill('Find a product','Freshona Lentils');b.screenshot('webshop_8509_owner.png')
    b.click('+ Add Product');b.wait('Product details →');b.screenshot('webshop_8509_add_product.png')
    b.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
    b.screenshot('webshop_8509_owner_mobile.png')
    mobile=dict(width=390,document_width=b.js('document.documentElement.scrollWidth'),viewport_width=b.js('window.innerWidth'))
    b.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
    if '--finish-existing' in sys.argv:
        print('Completed owner and employee screenshots after waiting for full page rendering.');b.ws.close();return
    b.click('Sign out');b.login('employee');b.wait('See it. Review it. Count it.')
    b.upload(Path(r'C:\Users\U938\Downloads\folder+\IMG_5034.jpeg'))
    b.wait('Check this photo');b.click('Check this photo');b.wait('Find products in this photo',120)
    b.click('Find products in this photo');b.wait('model proposals',240)
    b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Products in this photo'))?.scrollIntoView({block:'start'})")
    b.screenshot('webshop_8509_employee.png')
    b.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
    b.js("[...document.querySelectorAll('label')].find(e=>e.innerText.includes('Highlight a detection'))?.scrollIntoView({block:'start'})")
    b.screenshot('webshop_8509_employee_mobile.png')
    b.js("[...document.querySelectorAll('h3')].find(e=>e.innerText.includes('Products in this photo'))?.scrollIntoView({block:'start'})")
    b.screenshot('webshop_8509_employee_mobile_products.png')
    report=dict(actual_browser_screenshots=True,employee_inference='LIVE actual DINO and SigLIP2 on supplied scene',stock_action_clicked=False,mobile=mobile,
                employee_mobile=dict(document_width=b.js('document.documentElement.scrollWidth'),viewport_width=b.js('window.innerWidth')))
    (ROOT/'data/webshop_ux_demo/browser_visual_evidence.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));b.ws.close()


if __name__=='__main__':main()
