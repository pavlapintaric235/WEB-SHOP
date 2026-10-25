"""Create a narrated photo-free review video from actual sandbox screenshots.

Uses installed Windows speech and headless Chrome, no renderer/model download.
The result is explicitly a preview, not a claimed final submission or live run.
"""
import base64
import json
from pathlib import Path
import subprocess
import wave
from browser_evidence import Browser

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data/judge_release/video';folder.mkdir(parents=True,exist_ok=True)
    entries=[
      ('title','banner.svg','WEB&SHOP / InventoryLens','WEB and SHOP helps small shops turn product photos into reviewed inventory actions. This is a photo-free local interface preview. It does not claim that the public judge endpoint is deployed.'),
      ('owner','owner.png','Owner / one shared catalogue','Owners see one shop catalogue, prices, stock and registration status. This isolated judge shop uses fictional workbook data. Existing private photographs and model embeddings were not copied. Missing-photo panels mean references are still required.'),
      ('registration','registration.png','Register 3–5 useful views','The owner enters product identity, price, initial quantity and shelf, then supplies three to five useful views. OpenCV quality feedback and explicit approval precede registration. Approved images produce genuine reference embeddings without per-product retraining. This preview shows the form, not a newly completed registration.'),
      ('employee','employee.png','Employee / live processing is explicit','The employee workflow checks capture quality, calls the existing RunPod detector, ranks shop references with local SigLIP, and holds every suggestion for review. Private archived photos are disabled in the judge sandbox. Stock updates require a separate action and before-and-after confirmation. This start screen contains no inference results.'),
      ('buyer','buyer.png','Buyer / quantities and budget','Buyers choose quantities from the same shop catalogue and see estimated euro totals. Unpriced or unknown items remain explicit. A shopping list and a camera cart preview never modify inventory.'),
      ('map','buyer-map.png','Buyer / owner-assigned shelf stops','The map highlights saved shelves containing shopping-list items. Positions come from the fictional workbook and can be edited by the owner. Progress is manual. There is no simulated indoor position, and no shortest-route claim without a valid walkable grid.'),
      ('evidence','results.svg','Actual measured results / development evidence','Earlier genuine AWS execution returned OpenCV five quality evidence: proceed for a valid reference and retake for an intentionally blurred control. On ten actual product crops, batching reduced matching from fourteen point three seven to seven point five seven seconds, with unchanged SKU decisions. Prepared live recognition took twenty-seven point five nine seconds. These are bounded development observations, not broad accuracy or a new live execution in this video.'),
      ('finish','banner.svg','Review copy / final submission still pending','Partial products, unknowns and similar packages still require review. The final submission needs the real team introduction, approved footage of registration and inference, verified judge access, and a public or unlisted video upload. Credentials and private photos must remain protected.')]
    cues=[dict(id=i,image=image,title=title,narration=narration) for i,image,title,narration in entries]
    (folder/'cues.json').write_text(json.dumps(cues,indent=2))
    subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'tools/render_preview_audio.ps1')],check=True)
    combined=[];timeline=[];position=0;params=None
    for cue in cues:
        with wave.open(str(folder/(cue['id']+'.wav')),'rb') as source:
            if params is None:params=source.getparams()
            assert source.getnchannels()==params.nchannels and source.getframerate()==params.framerate
            audio=source.readframes(source.getnframes());duration=source.getnframes()/params.framerate
        silence=b'\0'*round(params.framerate*.5)*params.nchannels*params.sampwidth
        combined.extend([audio,silence])
        image_path=ROOT/'docs/public'/cue['image'];mime='image/svg+xml' if image_path.suffix=='.svg' else 'image/png'
        timeline.append(dict(start=position,end=position+duration+.5,title=cue['title'],image=f'data:{mime};base64,'+base64.b64encode(image_path.read_bytes()).decode()))
        position+=duration+.5
    with wave.open(str(folder/'narration.wav'),'wb') as target:
        target.setparams(params);target.writeframes(b''.join(combined))
    if position>290:raise RuntimeError('Preview narration exceeds the under-five-minute target.')
    audio_url='data:audio/wav;base64,'+base64.b64encode((folder/'narration.wav').read_bytes()).decode()
    b=Browser();b.call('Page.navigate',dict(url='about:blank'))
    script=r'''(async()=>{
      const cues=CUES,audioUrl=AUDIO;document.body.innerHTML='';document.body.style.margin='0';
      const canvas=document.createElement('canvas');canvas.width=1920;canvas.height=1080;document.body.append(canvas);const ctx=canvas.getContext('2d');
      const images=await Promise.all(cues.map(c=>new Promise((resolve,reject)=>{let i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=c.image;})));
      const audio=new AudioContext();await audio.resume();const buffer=await audio.decodeAudioData(await(await fetch(audioUrl)).arrayBuffer());
      const source=audio.createBufferSource();source.buffer=buffer;const dest=audio.createMediaStreamDestination();source.connect(dest);
      const stream=canvas.captureStream(10);stream.addTrack(dest.stream.getAudioTracks()[0]);
      const recorder=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp8,opus',videoBitsPerSecond:1200000});const chunks=[];
      const completed=new Promise(resolve=>{recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=async()=>{let reader=new FileReader();reader.onloadend=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(new Blob(chunks,{type:'video/webm'}));};});
      let start=audio.currentTime;
      const draw=()=>{const t=audio.currentTime-start;let n=cues.findIndex(c=>t>=c.start&&t<c.end);if(n<0)n=cues.length-1;const cue=cues[n],image=images[n];
       ctx.fillStyle='#102f29';ctx.fillRect(0,0,1920,1080);ctx.fillStyle='#afd8c4';ctx.font='22px Arial';ctx.fillText('WEB&SHOP / LOCAL REVIEW PREVIEW',50,45);ctx.fillStyle='white';ctx.font='bold 34px Arial';ctx.fillText(cue.title,50,100);
       const scale=Math.min(1820/image.width,820/image.height),w=image.width*scale,h=image.height*scale;ctx.drawImage(image,(1920-w)/2,130+(820-h)/2,w,h);
       ctx.fillStyle='#cce4d8';ctx.font='21px Arial';ctx.fillText('Actual sandbox screens · private photos omitted · no new live inference shown',50,1010);
       ctx.fillStyle='#d8ad64';ctx.fillRect(50,1040,1820*Math.min(1,t/buffer.duration),5);
      };
      draw();recorder.start();source.start();const interval=setInterval(draw,100);source.onended=()=>{clearInterval(interval);draw();recorder.stop();audio.close();};return await completed;
    })()'''.replace('CUES',json.dumps(timeline)).replace('AUDIO',json.dumps(audio_url))
    print(f'Rendering {position:.1f}-second narrated local preview; no publication.',flush=True)
    result=b.call('Runtime.evaluate',dict(expression=script,awaitPromise=True,returnByValue=True,userGesture=True))
    if 'exceptionDetails' in result:raise RuntimeError('Local video renderer failed.')
    output=folder/'webshop_judge_preview.webm';output.write_bytes(base64.b64decode(result['result']['value']))
    report=dict(path=str(output.relative_to(ROOT)),duration_seconds=round(position,2),bytes=output.stat().st_size,narration='Offline Windows speech',actual_ui_screenshots=True,private_photos_included=False,final_submission=False,
                pending=['Real team introduction','Approved registration/inference footage','Verified judge endpoint','Final video upload'])
    (folder/'video_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));b.ws.close()


if __name__=='__main__':main()
