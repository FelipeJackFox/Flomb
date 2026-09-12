import * as THREE from './dist/vendor/three.module.js';
import fs from 'node:fs';
const bytes=fs.readFileSync(new URL('./dist/assets/flybody.glb',import.meta.url));const d=JSON.parse(bytes.subarray(20,20+bytes.readUInt32LE(12)).toString());
const nodes=d.nodes.map(n=>{let o=new THREE.Object3D();o.name=n.name;o.position.fromArray(n.translation||[0,0,0]);o.quaternion.fromArray(n.rotation||[0,0,0,1]);return o});d.nodes.forEach((n,i)=>n.children?.forEach(c=>nodes[i].add(nodes[c])));
const anatomy=new THREE.Group();d.scenes[0].nodes.forEach(i=>anatomy.add(nodes[i]));anatomy.rotation.x=-Math.PI/2;
const fly=new THREE.Group();fly.add(anatomy);fly.rotation.y=Math.PI/2;fly.scale.setScalar(2.5);fly.position.set(0,1.40,.99);fly.updateMatrixWorld(true);
const {attachMouseLeg}=await import('data:text/javascript;base64,'+Buffer.from(fs.readFileSync(new URL('./dist/leg-ik.js',import.meta.url),'utf8').replace("from 'three'",'from '+JSON.stringify(new URL('./dist/vendor/three.module.js',import.meta.url).href))).toString('base64'));const solve=attachMouseLeg(fly);let maxError=0;for(let i=0;i<101;i++){const target=new THREE.Vector3(.23+Math.sin(i*.1)*.06,1.4114,.644+Math.cos(i*.1)*.05);maxError=Math.max(maxError,solve(target));}console.log('MAX_CONTACT_ERROR',maxError);if(maxError>.003)process.exit(1);
