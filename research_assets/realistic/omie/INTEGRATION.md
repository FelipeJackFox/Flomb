# Existing realistic computer assets

Author Omie’s Assets. Source and CC0 evidence: LICENSE-SOURCE.md and source-page.html.

Copy `glb/*.gltf`, `glb/*.bin`, `glb/*.png` together for 8.4 MB shared PBR textures. Alternative individual GLBs duplicate textures (32 MB for four). All dimensions are metres; +Y up, centered X/Z, base Y=0. Geometry preserved; normalization only translates origins.

| Model | Width X | Height Y | Depth Z | Orientation |
|---|---:|---:|---:|---|
| monitor | .745035 | .512370 | .145056 | front -Z, rotate Y pi for +Z |
| keyboard | .485026 | .023483 | .145384 | camera from +Z sees keys normally |
| mouse | .094754 | .058168 | .155606 | buttons point +Z, rotate Y pi toward -Z |
| computer | .172551 | .391450 | .369494 | front +Z |

With monitor rotation Y pi, display surface vertices in Three Y-up root coordinates (not additionally under rotated monitor):

- bottom-left [-.361303,.107517,-.006632]
- bottom-right [.361303,.107517,-.006632]
- top-right [.361303,.502808,-.028883]
- top-left [-.361303,.502808,-.028883]

Includes 1 mm offset to avoid z fighting. Equivalent plane center [0,.3051625,-.0177575], width .722606, height .395916, rotationX +.05623 (face +Z). Apply same scale and translation as monitor. Verified with offline Blender render qa-screen.png.

Mouse top maximum Y .058168, contact near button approximate [0,.045,-.03] after Y pi, must match chosen paw surface.

Offline QA: imported exported GLBs back into Blender; rendered actual textured models in qa.png and fitted display overlay in qa-screen.png. No browser verification claimed.

Blender official 4.5.0 ARM64 downloaded from https://download.blender.org/release/Blender4.5/blender-4.5.0-macos-arm64.dmg, mounted read-only /Volumes/Blender; no global installation. Conversion script convert.py.
