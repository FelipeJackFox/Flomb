import * as THREE from 'three';
// Presentation-only CCD: rotate the existing articulated leg, never stretch it.
export function attachMouseLeg(fly){
 const joints=['tibia_T1_right','femur_T1_right','coxa_T1_right'].map(n=>fly.getObjectByName(n));
 const tip=fly.getObjectByName('claw_T1_right');
 if(!tip||joints.some(j=>!j))throw Error('Falta la cadena de la pata delantera');
 const p=new THREE.Vector3(),end=new THREE.Vector3(),a=new THREE.Vector3(),b=new THREE.Vector3();
 const worldQ=new THREE.Quaternion(),parentQ=new THREE.Quaternion(),delta=new THREE.Quaternion();
 return target=>{
  for(let iteration=0;iteration<64;iteration++){
   for(const joint of joints){
    fly.updateMatrixWorld(true);joint.getWorldPosition(p);tip.getWorldPosition(end);
    a.copy(end).sub(p).normalize();b.copy(target).sub(p).normalize();
    delta.setFromUnitVectors(a,b);joint.getWorldQuaternion(worldQ);joint.parent.getWorldQuaternion(parentQ);
    joint.quaternion.copy(parentQ.invert().multiply(delta).multiply(worldQ)).normalize();
   }
   fly.updateMatrixWorld(true);if(tip.getWorldPosition(end).distanceTo(target)<.001)break;
  }
  return tip.getWorldPosition(end).distanceTo(target);
 };
}
export function cursorMotion(current,previous,size,phase){
 const t=Math.min(1,Math.max(0,phase/.55)),ease=t*t*(3-2*t);
 return {x:((previous%size)+(current%size-previous%size)*ease)/(size-1),y:(Math.floor(previous/size)+(Math.floor(current/size)-Math.floor(previous/size))*ease)/(size-1)};
}
