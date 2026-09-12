import * as THREE from 'three';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { attachMouseLeg,cursorMotion } from './leg-ik.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const $=id=>document.getElementById(id), loader=new GLTFLoader();
const state={playing:true,speed:1,index:0,replay:0,phase:0,last:performance.now(),ready:false};
let lastBrainStep=-1,solveMouseLeg;
const mouseContact=new THREE.Vector3();
let dataset,brainData,fly,mouse,screenTexture,brainPoints,brainEdges,stageView,brainView;
const screenCanvas=document.createElement('canvas');screenCanvas.width=1024;screenCanvas.height=640;
const screen=screenCanvas.getContext('2d');
const fmt=n=>new Intl.NumberFormat('es-MX').format(n);
function makeView(id,position,target){
 const element=$(id),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));renderer.outputColorSpace=THREE.SRGBColorSpace;
 renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.0;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 element.appendChild(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.01,100);
 camera.position.set(...position);const controls=new OrbitControls(camera,renderer.domElement);
 controls.target.set(...target);controls.enableDamping=true;controls.minDistance=1.6;controls.maxDistance=12;controls.maxPolarAngle=Math.PI*.87;
 const resize=()=>{const {width,height}=element.getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix()};
 new ResizeObserver(resize).observe(element);resize();controls.update();controls.saveState();
 return {scene,camera,controls,renderer};
}
async function model(name,scale,position,rotation=0){
 const files={desk:'metal_office_desk/metal_office_desk',chairDesk:'modern_arm_chair_01/modern_arm_chair_01',computerScreen:'omie/monitor',computerKeyboard:'omie/keyboard',computerMouse:'omie/mouse',computer:'omie/computer'};
 const path=`./assets/realistic/${files[name]}.gltf`;
 const {scene:asset}=await loader.loadAsync(path);
 const box=new THREE.Box3().setFromObject(asset),center=box.getCenter(new THREE.Vector3());
 asset.position.set(-center.x,-box.min.y,-center.z);
 const wrapper=new THREE.Group();wrapper.add(asset);wrapper.scale.setScalar(scale);wrapper.position.set(...position);wrapper.rotation.y=rotation;
 wrapper.traverse(n=>{if(n.isMesh){n.castShadow=true;n.receiveShadow=true}});stageView.scene.add(wrapper);return wrapper;
}
function setupLights(){
 const sc=stageView.scene;sc.add(new THREE.HemisphereLight(0xc8d5ff,0x27213b,.8));
 const key=new THREE.DirectionalLight(0xffe8cd,2.0);key.position.set(-3,6,4);key.castShadow=true;key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-4;key.shadow.camera.right=4;key.shadow.camera.top=4;key.shadow.camera.bottom=-4;key.shadow.normalBias=.015;sc.add(key);
 const rim=new THREE.DirectionalLight(0xcbd7ff,1.0);rim.position.set(2,3,-3);sc.add(rim);
 const fill=new THREE.PointLight(0x9ff6ed,4,6);fill.position.set(0,2,-.2);sc.add(fill);
 // Functional scene ground, not a replacement for any modeled asset.
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:0x161a29,roughness:1}));floor.receiveShadow=true;floor.rotation.x=-Math.PI/2;floor.position.y=-.02;sc.add(floor);
}
async function setupStage(){
 stageView=makeView('stage',[3.3,2.7,4.5],[0,1.15,.45]);setupLights();
 const hdr=await new HDRLoader().loadAsync('./assets/realistic/studio.hdr');const pmrem=new THREE.PMREMGenerator(stageView.renderer);stageView.scene.environment=pmrem.fromEquirectangular(hdr).texture;stageView.scene.environmentIntensity=.65;hdr.dispose();pmrem.dispose();
 const loaded=await Promise.all([model('desk',1.714,[0,0,0]),model('chairDesk',2.1,[0,0,1.45],Math.PI),model('computerScreen',1.5,[0,1.355,-.30],Math.PI),model('computerKeyboard',1.5,[-.25,1.355,.31]),model('computerMouse',1.2,[.23,1.355,.68],Math.PI),loader.loadAsync('./assets/flybody.glb')]);
 await model('computer',1.4,[1.12,1.355,-.22]);
 mouse=loaded[4];const anatomy=loaded[5].scene;anatomy.rotation.x=-Math.PI/2;
 fly=new THREE.Group();fly.add(anatomy);fly.rotation.y=Math.PI/2;fly.scale.setScalar(2.5);fly.position.set(0,1.40,.99);stageView.scene.add(fly);
 for(const side of ['left','right']){
  const wing=fly.getObjectByName('wing_'+side);if(wing){wing.userData.base=wing.quaternion.clone();wing.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),1.5)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.7)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),-1));}
 }
 fly.traverse(n=>{if(n.isMesh){n.castShadow=true;const materials=Array.isArray(n.material)?n.material:[n.material];for(const m of materials){m.side=THREE.DoubleSide;if(m.transparent)m.depthWrite=false;}}});
 screenTexture=new THREE.CanvasTexture(screenCanvas);screenTexture.colorSpace=THREE.SRGBColorSpace;
 const display=new THREE.Mesh(new THREE.PlaneGeometry(.722606*1.5,.395916*1.5),new THREE.MeshBasicMaterial({map:screenTexture}));display.position.set(0,1.355+.3051625*1.5,-.30-.0177575*1.5);display.rotation.x=.05623;stageView.scene.add(display);
 solveMouseLeg=attachMouseLeg(fly);
 $('asset-state').style.display='none';
}
function setupBrain(){
 brainView=makeView('brain',[3.7,1.2,4.0],[0,0,0]);brainView.controls.minDistance=1.7;
 const pos=new Float32Array(brainData.neurons.flatMap(n=>n.position));
 const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(pos.length).fill(.2),3));geo.setAttribute('strength',new THREE.BufferAttribute(new Float32Array(pos.length/3),1));
 brainPoints=new THREE.Points(geo,new THREE.ShaderMaterial({transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,vertexColors:true,uniforms:{pixelRatio:{value:Math.min(devicePixelRatio,1.7)}},vertexShader:`attribute float strength; varying vec3 tint; varying float intensity; uniform float pixelRatio; void main(){tint=color;intensity=strength;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=(3.0+23.0*sqrt(strength))*pixelRatio;}`,fragmentShader:`varying vec3 tint; varying float intensity; void main(){float d=length(gl_PointCoord-vec2(0.5))*2.0;if(d>1.0)discard;float halo=exp(-5.0*d*d);float core=1.0-smoothstep(0.08,0.32,d);gl_FragColor=vec4(mix(tint,vec3(1.0),core*0.65),max(0.08,intensity)*(halo*0.5+core*0.8));}`}));brainView.scene.add(brainPoints);
 const edgePositions=[];for(const [from,to] of brainData.edges){edgePositions.push(...brainData.neurons[from].position,...brainData.neurons[to].position)}
 const eg=new THREE.BufferGeometry();eg.setAttribute('position',new THREE.Float32BufferAttribute(edgePositions,3));eg.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(edgePositions.length).fill(.1),3));
 brainEdges=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.12,depthWrite:false}));brainView.scene.add(brainEdges);
 $('brain-note').textContent=`${fmt(brainData.sample_count)} de ${fmt(brainData.total_neurons)} neuronas · ${fmt(brainData.edges.length)} conexiones visibles`;
}
const replay=()=>dataset.replays[state.replay], frame=()=>replay().frames[state.index];
function refreshBrain(){
 if(!brainPoints)return;
 const steps=frame().activity_steps||[frame().activity];
 const step=Math.min(steps.length-1,Math.floor(state.phase/.64*steps.length));lastBrainStep=step;
 const h=steps[step],scale=dataset.activity_scale||1,colors=brainPoints.geometry.attributes.color, strengths=brainPoints.geometry.attributes.strength;
 const neg=new THREE.Color('#c089ff'),pos=new THREE.Color('#58ffe0'),dim=new THREE.Color('#11192b');
 let peak=0;
 h.forEach((a,i)=>{peak=Math.max(peak,Math.abs(a));const strength=Math.min(1,Math.abs(a)/scale);const c=dim.clone().lerp(a<0?neg:pos,Math.sqrt(strength));colors.setXYZ(i,c.r,c.g,c.b);strengths.setX(i,strength)});colors.needsUpdate=true;strengths.needsUpdate=true;
 const ec=brainEdges.geometry.attributes.color;
 brainData.edges.forEach(([from,to,weight],i)=>{const strength=Math.sqrt(Math.min(1,Math.abs(h[from])/scale));const color=dim.clone().lerp(weight<0?neg:pos,strength);ec.setXYZ(i*2,color.r,color.g,color.b);ec.setXYZ(i*2+1,color.r,color.g,color.b)});ec.needsUpdate=true;
 $('neural-step').textContent=`Cálculo ${step+1} / ${steps.length} · pico |actividad| ${peak.toFixed(3)}`;
}
function updateBoard(){
 const r=replay(),f=frame(),revealed=state.phase>=.64,visible=revealed?f.after:f.visible;
 $('board').style.gridTemplateColumns=`repeat(${r.size},1fr)`;$('board').style.gap=r.size>9?'2px':'4px';
 $('board').replaceChildren(...visible.map((n,i)=>{const cell=document.createElement('span');cell.className=`cell ${n>=0?'open n'+n:''} ${i===f.action?'chosen':''} ${revealed&&f.outcome==='mine'&&i===f.action?'mine':''}`;cell.textContent=revealed&&f.outcome==='mine'&&i===f.action?'✹':n>0?String(n):'';return cell}));
 $('mine-count').textContent=String(r.mines).padStart(3,'0');$('classic-face').textContent=revealed&&f.outcome==='mine'?'☹':'☺';
 $('decision').textContent=`Clic: fila ${Math.floor(f.action/r.size)+1}, columna ${f.action%r.size+1}`;
 $('probability').textContent=`P(acción) ${(f.probability*100).toFixed(1)}%`;
 $('outcome').textContent=revealed&&f.outcome==='mine'?'Encontró una mina':revealed&&f.outcome==='win'?'Tablero resuelto':'Observando el tablero';
 $('game-label').textContent=`Partida ${state.replay+1} de ${dataset.replays.length} · semilla ${r.seed}`;
 $('step-label').textContent=`${state.index+1} / ${r.frames.length}`;$('seek').max=r.frames.length-1;$('seek').value=state.index;
}
function paintScreen(){
 if(!dataset)return;const r=replay(),f=frame(),revealed=state.phase>=.64,v=revealed?f.after:f.visible;
 screen.fillStyle='#008080';screen.fillRect(0,0,1024,640);
 const ox=262,oy=120,pitch=460/r.size;
 function bevel(x,y,w,h,raised=true){screen.fillStyle='#c0c0c0';screen.fillRect(x,y,w,h);screen.strokeStyle=raised?'#fff':'#808080';screen.lineWidth=3;screen.beginPath();screen.moveTo(x,y+h);screen.lineTo(x,y);screen.lineTo(x+w,y);screen.stroke();screen.strokeStyle=raised?'#808080':'#fff';screen.beginPath();screen.moveTo(x+w,y);screen.lineTo(x+w,y+h);screen.lineTo(x,y+h);screen.stroke();}
 bevel(242,12,500,596);screen.fillStyle='#000080';screen.fillRect(247,17,490,30);screen.fillStyle='#fff';screen.font='bold 22px Arial';screen.fillText('Buscaminas',257,40);
 screen.fillStyle='#000';screen.font='16px Arial';screen.fillText('Juego    Ayuda',255,67);bevel(257,77,470,34,false);
 const seconds=Math.min(999,Math.floor(state.index*5.2+state.phase*4));$('classic-clock').textContent=String(seconds).padStart(3,'0');
 screen.fillStyle='#180000';screen.fillRect(264,80,62,29);screen.fillRect(657,80,62,29);screen.fillStyle='#f11';screen.font='bold 26px monospace';screen.fillText(String(r.mines).padStart(3,'0'),268,103);screen.fillText(String(seconds).padStart(3,'0'),661,103);
 bevel(477,79,31,30);screen.fillStyle='#ffeb00';screen.fillRect(481,82,23,23);screen.fillStyle='#000';screen.font='27px Arial';screen.fillText(revealed&&f.outcome==='mine'?'☹':'☺',479,103);
 const numberColors=['','#0000ff','#008000','#ff0000','#000080','#800000','#008080','#000','#808080'];
 v.forEach((n,i)=>{const x=ox+(i%r.size)*pitch,y=oy+Math.floor(i/r.size)*pitch;
  if(n<0)bevel(x,y,pitch,pitch);else{screen.fillStyle='#c0c0c0';screen.fillRect(x,y,pitch,pitch);screen.strokeStyle='#888';screen.lineWidth=1;screen.strokeRect(x,y,pitch,pitch);}
  if(i===f.action&&revealed&&f.outcome==='mine'){screen.fillStyle='#f00';screen.fillRect(x,y,pitch,pitch);screen.fillStyle='#000';screen.font=`bold ${pitch*.85}px Arial`;screen.textAlign='center';screen.fillText('✹',x+pitch/2,y+pitch*.82);}
  else if(n>0){screen.fillStyle=numberColors[n];screen.font=`bold ${pitch*.8}px Arial`;screen.textAlign='center';screen.fillText(n,x+pitch/2,y+pitch*.8);}screen.textAlign='left';
 });
 const previous=state.index?r.frames[state.index-1].action:Math.floor(r.size/2)*r.size+Math.floor(r.size/2);
 const motion=cursorMotion(f.action,previous,r.size,state.phase);
 const cx=ox+(motion.x*(r.size-1)+.5)*pitch,cy=oy+(motion.y*(r.size-1)+.5)*pitch;
 screen.beginPath();screen.moveTo(cx,cy);screen.lineTo(cx+7,cy+26);screen.lineTo(cx+13,cy+18);screen.lineTo(cx+25,cy+19);screen.closePath();screen.fillStyle='#fff';screen.fill();screen.strokeStyle='#000';screen.lineWidth=2;screen.stroke();
 if(screenTexture)screenTexture.needsUpdate=true;
}
function advance(){
 if(state.index+1<replay().frames.length)state.index++;
 else if(state.replay+1<dataset.replays.length){state.replay++;state.index=0;$('replay-select').value=state.replay;}
 else{state.playing=false;state.phase=1.3;$('play').textContent='Volver al inicio';$('outcome').textContent='Fin de las partidas grabadas';return;}
 state.phase=0;refreshBrain();updateBoard();
}
function animate(now){
 requestAnimationFrame(animate);const delta=Math.min(.1,(now-state.last)/1000);state.last=now;
 if(state.ready){
  const prev=state.phase>=.64;
  if(state.playing){state.phase+=delta*state.speed/4;if(state.phase>=1.3)advance()}
  if(prev!==(state.phase>=.64))updateBoard();
  const count=frame().activity_steps?.length||1;if(Math.min(count-1,Math.floor(state.phase/.64*count))!==lastBrainStep)refreshBrain();paintScreen();
  if(mouse){
   const r=replay(),previous=state.index?r.frames[state.index-1].action:Math.floor(r.size/2)*r.size+Math.floor(r.size/2);
   const cursor=cursorMotion(frame().action,previous,r.size,state.phase);
   mouse.position.x=.23+(cursor.x-.5)*.12;mouse.position.z=.68+(cursor.y-.5)*.10;
   mouse.rotation.x=state.phase>.6&&state.phase<.72?-.025:0;
   mouse.updateMatrixWorld(true);mouseContact.set(0,.047,.03);mouse.localToWorld(mouseContact);
   if(solveMouseLeg)solveMouseLeg(mouseContact);
  }
 }
 for(const v of [stageView,brainView])if(v){v.controls.update();v.renderer.render(v.scene,v.camera)}
}
async function refreshStatus(){
 let data,live=false;
 try{const r=await fetch('./api/status',{cache:'no-store'});if(!r.ok)throw Error();data=await r.json();live=true}catch{data=await (await fetch('./data/status-snapshot.json')).json()}
 $('episodes').textContent=fmt(data.episode);$('progress').value=data.episode;$('percentage').textContent=(data.episode/data.target*100).toFixed(1)+'%';$('mix').textContent=data.mixture.map(x=>Math.round(x*100)+'%').join(' / ');$('memory').textContent=(data.peak_rss_bytes_macos/1e9).toFixed(2)+' GB';$('live-label').textContent=live?'Estado del entrenamiento local':'Estado guardado · reproducción web';
}
$('play').onclick=()=>{if(state.replay===dataset.replays.length-1&&state.index===replay().frames.length-1&&state.phase>=1.3){state.replay=0;state.index=0;state.phase=0;$('replay-select').value=0;refreshBrain();updateBoard();}state.playing=!state.playing;$('play').textContent=state.playing?'Pausar':'Reproducir'};
$('next').onclick=()=>{advance()};$('speed').onchange=e=>state.speed=Number(e.target.value);
$('seek').oninput=e=>{state.index=Number(e.target.value);state.phase=0;refreshBrain();updateBoard()};
$('replay-select').onchange=e=>{state.replay=Number(e.target.value);state.index=0;state.phase=0;refreshBrain();updateBoard()};
$('edges').onchange=e=>{if(brainEdges)brainEdges.visible=e.target.checked};$('brain-home').onclick=()=>brainView.controls.reset();
$('credits-button').onclick=()=>$('credits').showModal();$('close-credits').onclick=()=>$('credits').close();
requestAnimationFrame(animate);
try{
 [dataset,brainData]=await Promise.all(['replays','brain'].map(async x=>{const r=await fetch(`./data/${x}.json`);if(!r.ok)throw Error('No se pudo cargar '+x);return r.json()}));
 $('replay-select').replaceChildren(...dataset.replays.map((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.size}×${r.size} · ${r.frames.length} clics`;return o}));
 $('checkpoint').textContent=`Checkpoint ${fmt(dataset.episode)} · decisiones y actividad grabadas`;
 $('replay-note').textContent=`${dataset.replays.length} partidas de semillas prefijadas, sin elegir victorias. Se reproducen una sola vez. No es juego en vivo.`;
 setupBrain();await setupStage();state.ready=true;refreshBrain();updateBoard();await refreshStatus();
 setInterval(()=>{if(!document.hidden)refreshStatus().catch(()=>{})},30000);
 const tools=document.modelContext;if(tools?.registerTool){tools.registerTool({name:'set_replay',description:'Selecciona una reproducción real y una decisión; no altera el entrenamiento.',inputSchema:{type:'object',properties:{replay:{type:'integer',minimum:0},step:{type:'integer',minimum:0}},required:['replay','step'],additionalProperties:false},execute:({replay:r,step})=>{if(!Number.isInteger(r)||!Number.isInteger(step)||!dataset.replays[r]?.frames[step])throw Error('Reproducción o decisión inválida');state.replay=r;state.index=step;state.phase=0;$('replay-select').value=r;refreshBrain();updateBoard();return {replay:r,step,checkpoint:dataset.checkpoint}},annotations:{readOnlyHint:false}})}
}catch(error){$('asset-state').style.display='grid';$('asset-state').textContent='No se pudo abrir el visor: '+error.message;console.error(error)}
