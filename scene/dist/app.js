import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const $=id=>document.getElementById(id), loader=new GLTFLoader();
const state={playing:true,speed:1,index:0,replay:0,phase:0,last:performance.now(),ready:false};
let dataset,brainData,fly,mouse,screenTexture,brainPoints,brainEdges,stageView,brainView;
const screenCanvas=document.createElement('canvas');screenCanvas.width=1024;screenCanvas.height=640;
const screen=screenCanvas.getContext('2d');
const fmt=n=>new Intl.NumberFormat('es-MX').format(n);
function makeView(id,position,target){
 const element=$(id),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));renderer.outputColorSpace=THREE.SRGBColorSpace;
 renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
 element.appendChild(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.01,100);
 camera.position.set(...position);const controls=new OrbitControls(camera,renderer.domElement);
 controls.target.set(...target);controls.enableDamping=true;controls.minDistance=1.6;controls.maxDistance=12;controls.maxPolarAngle=Math.PI*.87;
 const resize=()=>{const {width,height}=element.getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix()};
 new ResizeObserver(resize).observe(element);resize();controls.update();controls.saveState();
 return {scene,camera,controls,renderer};
}
async function model(name,scale,position,rotation=0){
 const {scene:asset}=await loader.loadAsync(`./assets/furniture/${name}.glb`);
 const box=new THREE.Box3().setFromObject(asset),center=box.getCenter(new THREE.Vector3());
 asset.position.set(-center.x,-box.min.y,-center.z);
 const wrapper=new THREE.Group();wrapper.add(asset);wrapper.scale.setScalar(scale);wrapper.position.set(...position);wrapper.rotation.y=rotation;
 wrapper.traverse(n=>{if(n.isMesh){n.castShadow=true;n.receiveShadow=true}});stageView.scene.add(wrapper);return wrapper;
}
function setupLights(){
 const sc=stageView.scene;sc.add(new THREE.HemisphereLight(0xc8d5ff,0x27213b,2.5));
 const key=new THREE.DirectionalLight(0xffe8cd,3.5);key.position.set(-3,6,4);sc.add(key);
 const rim=new THREE.DirectionalLight(0x9783ff,2.5);rim.position.set(2,3,-3);sc.add(rim);
 const fill=new THREE.PointLight(0x9ff6ed,4,6);fill.position.set(0,2,-.2);sc.add(fill);
 // Functional scene ground, not a replacement for any modeled asset.
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x161a29,roughness:1}));floor.rotation.x=-Math.PI/2;floor.position.y=-.02;sc.add(floor);
}
async function setupStage(){
 stageView=makeView('stage',[3.3,2.6,4.0],[0,1.18,.3]);setupLights();
 const loaded=await Promise.all([model('desk',3.5,[0,0,0]),model('chairDesk',2.65,[0,0,1.06],Math.PI),model('computerScreen',2.7,[0,1.35,-.36]),model('computerKeyboard',2.8,[-.20,1.355,.29]),model('computerMouse',2.5,[.43,1.355,.51]),loader.loadAsync('./assets/flybody.glb')]);
 mouse=loaded[4];const anatomy=loaded[5].scene;anatomy.rotation.x=-Math.PI/2;
 fly=new THREE.Group();fly.add(anatomy);fly.rotation.y=Math.PI/2;fly.scale.setScalar(2.5);fly.position.set(0,1.40,.99);stageView.scene.add(fly);
 for(const side of ['left','right']){
  const wing=fly.getObjectByName('wing_'+side);if(wing){wing.userData.base=wing.quaternion.clone();wing.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),1.5)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.7)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),-1));}
 }
 fly.traverse(n=>{if(n.isMesh){n.castShadow=true;const materials=Array.isArray(n.material)?n.material:[n.material];for(const m of materials){m.side=THREE.DoubleSide;if(m.transparent)m.depthWrite=false;}}});
 screenTexture=new THREE.CanvasTexture(screenCanvas);screenTexture.colorSpace=THREE.SRGBColorSpace;
 const display=new THREE.Mesh(new THREE.PlaneGeometry(.9,.49),new THREE.MeshBasicMaterial({map:screenTexture}));display.position.set(0,1.85,-.216);stageView.scene.add(display);
 $('asset-state').style.display='none';
}
function setupBrain(){
 brainView=makeView('brain',[3.7,1.2,4.0],[0,0,0]);brainView.controls.minDistance=1.7;
 const pos=new Float32Array(brainData.neurons.flatMap(n=>n.position));
 const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(pos.length).fill(.2),3));
 brainPoints=new THREE.Points(geo,new THREE.PointsMaterial({size:.021,vertexColors:true,transparent:true,opacity:.95,sizeAttenuation:true}));brainView.scene.add(brainPoints);
 const edgePositions=[];for(const [from,to] of brainData.edges){edgePositions.push(...brainData.neurons[from].position,...brainData.neurons[to].position)}
 const eg=new THREE.BufferGeometry();eg.setAttribute('position',new THREE.Float32BufferAttribute(edgePositions,3));eg.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(edgePositions.length).fill(.1),3));
 brainEdges=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.21,depthWrite:false}));brainView.scene.add(brainEdges);
 $('brain-note').textContent=`${fmt(brainData.sample_count)} de ${fmt(brainData.total_neurons)} neuronas · ${fmt(brainData.edges.length)} conexiones visibles`;
}
const replay=()=>dataset.replays[state.replay], frame=()=>replay().frames[state.index];
function refreshBrain(){
 if(!brainPoints)return;
 const h=frame().activity,scale=Math.max(.002,...h.map(Math.abs)),colors=brainPoints.geometry.attributes.color;
 const neg=new THREE.Color('#aa85ff'),pos=new THREE.Color('#7bf8de'),dim=new THREE.Color('#252c45');
 h.forEach((a,i)=>{const c=dim.clone().lerp(a<0?neg:pos,Math.pow(Math.abs(a)/scale,.55));colors.setXYZ(i,c.r,c.g,c.b)});colors.needsUpdate=true;
 const ec=brainEdges.geometry.attributes.color;
 brainData.edges.forEach(([from,to,weight],i)=>{const strength=Math.pow(Math.abs(h[from])/scale,.6);const color=dim.clone().lerp(weight<0?neg:pos,strength);ec.setXYZ(i*2,color.r,color.g,color.b);ec.setXYZ(i*2+1,color.r,color.g,color.b)});ec.needsUpdate=true;
}
function updateBoard(){
 const r=replay(),f=frame(),revealed=state.phase>=.64,visible=revealed?f.after:f.visible;
 $('board').style.gridTemplateColumns=`repeat(${r.size},1fr)`;$('board').style.gap=r.size>9?'2px':'4px';
 $('board').replaceChildren(...visible.map((n,i)=>{const cell=document.createElement('span');cell.className=`cell ${n>=0?'open n'+n:''} ${i===f.action?'chosen':''} ${revealed&&f.outcome==='mine'&&i===f.action?'mine':''}`;cell.textContent=revealed&&f.outcome==='mine'&&i===f.action?'×':n>0?String(n):'';return cell}));
 $('decision').textContent=`Clic: fila ${Math.floor(f.action/r.size)+1}, columna ${f.action%r.size+1}`;
 $('probability').textContent=`P(acción) ${(f.probability*100).toFixed(1)}%`;
 $('outcome').textContent=revealed&&f.outcome==='mine'?'Encontró una mina':revealed&&f.outcome==='win'?'Tablero resuelto':'Observando el tablero';
 $('step-label').textContent=`${state.index+1} / ${r.frames.length}`;$('seek').max=r.frames.length-1;$('seek').value=state.index;
}
function paintScreen(){
 if(!dataset)return;const r=replay(),f=frame(),revealed=state.phase>=.64,v=revealed?f.after:f.visible;
 screen.fillStyle='#101522';screen.fillRect(0,0,1024,640);screen.fillStyle='#99abd7';screen.font='26px sans-serif';screen.fillText(`BUSCAMINAS   ${r.size} × ${r.size}   /   ${r.mines} minas`,40,48);
 const pitch=490/r.size,ox=(1024-490)/2,oy=100;
 v.forEach((n,i)=>{const x=ox+(i%r.size)*pitch,y=oy+Math.floor(i/r.size)*pitch;screen.fillStyle=n<0?'#33405d':'#172338';if(i===f.action)screen.fillStyle=revealed&&f.outcome==='mine'?'#824960':'#6b54a0';screen.fillRect(x+1,y+1,pitch-2,pitch-2);if(n>0){screen.fillStyle=['','#80b4ff','#6de0bd','#e9aa80','#b597ee'][n]||'#d4c5ff';screen.font=`bold ${pitch*.62}px sans-serif`;screen.textAlign='center';screen.fillText(n,x+pitch/2,y+pitch*.72);screen.textAlign='left'}});
 const previous=state.index?replay().frames[state.index-1].action:Math.floor(r.size*r.size/2);
 const t=Math.min(1,state.phase/.55),ease=t*t*(3-2*t);
 const x0=previous%r.size,y0=Math.floor(previous/r.size),x1=f.action%r.size,y1=Math.floor(f.action/r.size);
 const cx=ox+(x0+(x1-x0)*ease+.5)*pitch,cy=oy+(y0+(y1-y0)*ease+.5)*pitch;
 screen.beginPath();screen.moveTo(cx,cy);screen.lineTo(cx+7,cy+26);screen.lineTo(cx+13,cy+18);screen.lineTo(cx+25,cy+19);screen.closePath();screen.fillStyle='#fff';screen.fill();screen.strokeStyle='#181326';screen.lineWidth=2;screen.stroke();
 if(screenTexture)screenTexture.needsUpdate=true;
}
function advance(){state.index=(state.index+1)%replay().frames.length;state.phase=0;refreshBrain();updateBoard()}
function animate(now){
 requestAnimationFrame(animate);const delta=Math.min(.1,(now-state.last)/1000);state.last=now;
 if(state.ready){
  const prev=state.phase>=.64;
  if(state.playing){state.phase+=delta*state.speed/2.5;if(state.phase>=1.3)advance()}
  if(prev!==(state.phase>=.64))updateBoard();paintScreen();
  if(mouse){const r=replay(),a=frame().action;mouse.position.x=.43+((a%r.size)/(r.size-1)-.5)*.10;mouse.position.z=.51+(Math.floor(a/r.size)/(r.size-1)-.5)*.08;mouse.rotation.x=state.phase>.6&&state.phase<.72?-.06:0;}
  if(fly){const joint=fly.getObjectByName('coxa_T1_right');if(joint){if(!joint.userData.home)joint.userData.home=joint.quaternion.clone();joint.quaternion.copy(joint.userData.home).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.15*Math.sin(state.phase*Math.PI)));}}
 }
 for(const v of [stageView,brainView])if(v){v.controls.update();v.renderer.render(v.scene,v.camera)}
}
async function refreshStatus(){
 let data,live=false;
 try{const r=await fetch('./api/status',{cache:'no-store'});if(!r.ok)throw Error();data=await r.json();live=true}catch{data=await (await fetch('./data/status-snapshot.json')).json()}
 $('episodes').textContent=fmt(data.episode);$('progress').value=data.episode;$('percentage').textContent=(data.episode/data.target*100).toFixed(1)+'%';$('mix').textContent=data.mixture.map(x=>Math.round(x*100)+'%').join(' / ');$('memory').textContent=(data.peak_rss_bytes_macos/1e9).toFixed(2)+' GB';$('live-label').textContent=live?'Estado del entrenamiento local':'Estado guardado · reproducción web';
}
$('play').onclick=()=>{state.playing=!state.playing;$('play').textContent=state.playing?'Pausar':'Reproducir'};
$('next').onclick=()=>{advance();state.phase=.66;updateBoard()};$('speed').onchange=e=>state.speed=Number(e.target.value);
$('seek').oninput=e=>{state.index=Number(e.target.value);state.phase=0;refreshBrain();updateBoard()};
$('replay-select').onchange=e=>{state.replay=Number(e.target.value);state.index=0;state.phase=0;refreshBrain();updateBoard()};
$('edges').onchange=e=>{if(brainEdges)brainEdges.visible=e.target.checked};$('brain-home').onclick=()=>brainView.controls.reset();
$('credits-button').onclick=()=>$('credits').showModal();$('close-credits').onclick=()=>$('credits').close();
requestAnimationFrame(animate);
try{
 [dataset,brainData]=await Promise.all(['replays','brain'].map(async x=>{const r=await fetch(`./data/${x}.json`);if(!r.ok)throw Error('No se pudo cargar '+x);return r.json()}));
 $('replay-select').replaceChildren(...dataset.replays.map((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${r.size}×${r.size} · ${r.mines} minas`;return o}));
 $('checkpoint').textContent=`Checkpoint ${fmt(dataset.episode)} · decisiones y actividad grabadas`;
 setupBrain();await setupStage();state.ready=true;refreshBrain();updateBoard();await refreshStatus();
 setInterval(()=>{if(!document.hidden)refreshStatus().catch(()=>{})},30000);
 const tools=document.modelContext;if(tools?.registerTool){tools.registerTool({name:'set_replay',description:'Selecciona una reproducción real y una decisión; no altera el entrenamiento.',inputSchema:{type:'object',properties:{replay:{type:'integer',minimum:0},step:{type:'integer',minimum:0}},required:['replay','step'],additionalProperties:false},execute:({replay:r,step})=>{if(!Number.isInteger(r)||!Number.isInteger(step)||!dataset.replays[r]?.frames[step])throw Error('Reproducción o decisión inválida');state.replay=r;state.index=step;state.phase=0;$('replay-select').value=r;refreshBrain();updateBoard();return {replay:r,step,checkpoint:dataset.checkpoint}},annotations:{readOnlyHint:false}})}
}catch(error){$('asset-state').style.display='grid';$('asset-state').textContent='No se pudo abrir el visor: '+error.message;console.error(error)}
