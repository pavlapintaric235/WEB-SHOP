"""Small CDP helper for an isolated browser; optional websockets 17.2 tooling."""
import base64
import json
from pathlib import Path
import time
from urllib.request import urlopen
from websockets.sync.client import connect


class Browser:
    def __init__(self,port=9231,url='http://localhost:8511'):
        pages=json.load(urlopen(f'http://127.0.0.1:{port}/json'))
        page=next(p for p in pages if p.get('url','').startswith(url))
        self.ws=connect(page['webSocketDebuggerUrl'],legacy=True,max_size=64*1024*1024);self.identity=0

    def call(self,method,params=None):
        self.identity+=1;identity=self.identity
        self.ws.send(json.dumps(dict(id=identity,method=method,params=params or {})))
        while True:
            result=json.loads(self.ws.recv())
            if result.get('id')==identity:
                if 'error' in result:raise RuntimeError(result['error'])
                return result.get('result',{})

    def js(self,expression):
        result=self.call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        if 'exceptionDetails' in result:raise RuntimeError('Browser script failed; inspect the isolated local page.')
        return result.get('result',{}).get('value')

    def wait(self,text,timeout=90):
        start=time.monotonic()
        while time.monotonic()-start<timeout:
            if text in self.js('document.body.innerText'):return
            time.sleep(.3)
        raise RuntimeError('UI did not show: '+text)

    def idle(self,timeout=120):
        start=time.monotonic();time.sleep(.4)
        while time.monotonic()-start<timeout:
            if not self.js("document.body.innerText.split('\\n').some(t=>t.trim()==='Stop')"):return
            time.sleep(.3)
        raise RuntimeError('App operation did not finish')

    def click(self,label):
        self.idle()
        self.js("(()=>{let e=[...document.querySelectorAll('button')].find(e=>e.innerText.trim()==="+json.dumps(label)+");if(!e)throw Error('Missing button');e.click()})()")
        time.sleep(.5)

    def radio(self,label):
        self.idle()
        self.js("(()=>{let e=[...document.querySelectorAll('label')].find(e=>e.innerText.trim()==="+json.dumps(label)+");if(!e)throw Error('Missing choice');e.click()})()")
        time.sleep(.7)

    def fill(self,label,value):
        self.idle();self.wait(label)
        query="[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')==="+json.dumps(label)+")"
        start=time.monotonic()
        while not self.js('Boolean('+query+')'):
            if time.monotonic()-start>60:raise RuntimeError('Input did not render')
            time.sleep(.3)
        self.js(query+'.focus()')
        self.call('Input.dispatchKeyEvent',dict(type='rawKeyDown',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=2))
        self.call('Input.dispatchKeyEvent',dict(type='keyUp',key='a',code='KeyA',windowsVirtualKeyCode=65,modifiers=0))
        self.call('Input.insertText',dict(text=str(value)))
        if label not in ('Username','Password','Judge access code'):
            self.call('Input.dispatchKeyEvent',dict(type='keyDown',key='Enter',code='Enter',windowsVirtualKeyCode=13,modifiers=0,text='\r'))
            self.call('Input.dispatchKeyEvent',dict(type='keyUp',key='Enter',code='Enter',windowsVirtualKeyCode=13,modifiers=0))
        self.js('document.activeElement.blur()');time.sleep(.7);self.idle()

    def viewport(self,width=1440,height=1000):
        self.call('Emulation.setDeviceMetricsOverride',dict(width=width,height=height,deviceScaleFactor=1,mobile=width<760))

    def screenshot(self,path):
        self.idle();self.js("window.getSelection()?.removeAllRanges();[...document.querySelectorAll('button')].find(e=>e.innerText.trim()===\"Don't show again\")?.click()")
        time.sleep(.5)
        data=self.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(base64.b64decode(data))
