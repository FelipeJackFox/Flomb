# Assets existentes para la escena de Buscaminas

Investigación y descarga: 2026-09-12. No se generó ninguna anatomía ni mueble. No se compraron assets. La carpeta es investigación independiente, no un Site ni un proyecto web inicializado.

## Selección utilizable

| Recurso | Archivos locales | Fuente y licencia |
|---|---|---|
| Drosophila anatómica Flybody | `flybody/flybody.glb`, `flybody/fruitfly.xml`, `flybody/LICENSE` | [Google DeepMind MuJoCo Menagerie / Flybody](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/flybody), Apache-2.0; autores Google DeepMind + HHMI Janelia. Fuente original [TuragaLab/flybody](https://github.com/TuragaLab/flybody). |
| Escritorio | `kenney/desk.glb` | [Kenney Furniture Kit](https://kenney.nl/assets/furniture-kit), CC0, licencia copiada en `kenney/License.txt` |
| Silla de escritorio | `kenney/chairDesk.glb` | Mismo pack y licencia |
| Monitor | `kenney/computerScreen.glb` | Mismo pack y licencia |
| Mouse | `kenney/computerMouse.glb` | Mismo pack y licencia |
| Teclado | `kenney/computerKeyboard.glb` | Mismo pack y licencia |
| Planta | `kenney/plantSmall1.glb` | Mismo pack y licencia |
| 140 muebles/decoraciones completos | `kenney/catalog/`, índice `kenney/catalog.json` | Mismo pack; hay lámparas, alfombras, libros, paredes, sillas y mesas alternativas |
| Iluminación HDRI estudio | `polyhaven/poly_haven_studio_1k.hdr` | [Poly Haven Studio](https://polyhaven.com/a/poly_haven_studio), Greg Zaal, CC0. URL/hash guardados en SOURCE.json. |

## Conversión de mosca, orientación y animación

`convert_flybody.py` convierte los 85 OBJ existentes usando posiciones, cuaterniones y jerarquía de cuerpos de `fruitfly.xml`. Aplica la escala 0.1 definida por MJCF. Triángulos y vértices originales se conservan; las normales se calculan de sus caras. Materiales y transparencia provienen de MJCF. No se ejecutó física ni se entrenó control motor. La pose corresponde a qpos=0.

GLB: 22,987,712 bytes, 85 mallas, 153 nodos, 272,550 triángulos. Bounds locales: mínimo (-0.19268, -0.30581, -0.13188), máximo (0.10854, 0.30581, 0.05369). Z es arriba; X apunta hacia la cabeza; Y es lateral. En Three.js, una rotación de raíz X=-PI/2 convierte a Y arriba. Después puede orientarse la raíz para mirar al monitor. El origen está en el tórax, no en los pies.

Conservar cada quaternion local como pose base. Articular nodos de cuerpo (no `mesh_...`) mantiene descendientes unidos: `head`, `wing_left`, `wing_right`, `coxa_T1_right`, `femur_T1_right`, `tibia_T1_right`, cadenas `tarsus_*`. T1 son patas delanteras, T2 medias, T3 traseras. Puede plegarse el cuerpo/patas para sentarla y mover una pata con el mouse como animación cinematográfica. No afirmar que la política aprende control motor.

La imagen oficial `flybody/flybody.png` fue inspeccionada: ojos compuestos rojos, seis patas segmentadas, dos alas con venación, antenas/aristas, halterios y abdomen reconocibles. Se inspeccionaron también los previews originales Kenney de monitor y silla. La conversión GLB necesita la comprobación visual final del integrador en su visor. La inspección de imagen oficial no demuestra por sí sola que el GLB convertido se vea correctamente.

El pack Kenney trae GLB autocontenidos, geometría de baja complejidad y materiales originales; no requiere modelar monitor, silla, mesa ni mouse. Índice de bounds por malla en catalog.json. El integrador debe usar Box3 después de cargar, porque los nodos también tienen transformaciones.

## Alternativas investigadas

- [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym/): alternativa científica basada en micro-CT de una hembra, con locomoción/sensores; Apache-2.0 según publicación. No seleccionada porque Flybody ya ofrece piezas detalladas, materiales y jerarquía descargables sin instalar el simulador. No debe presentarse como cuerpo reconstruido del mismo individuo MaleCNS.
- [NeuroMechFly original](https://github.com/NeLy-EPFL/NeuroMechFly): repositorio archivado, modelo legado; no seleccionado frente a las alternativas actuales.
- [Flyplotlib](https://github.com/tkclam/flyplotlib): integración gráfica de meshes/rig de NeuroMechFly; alternativa de inspección científica, no necesaria para cargar el GLB.
- [Drosophila anatomical model / joaquinvilla, Birmingham](https://sketchfab.com/3d-models/3d-drosophila-fly-model-22b1ccca937a421da16f511a09042291): 1.4 millones de triángulos, impresión anatómica; no se confirmó permiso de descarga/reutilización y no se descargó.
- [Poly Pizza Furniture Kit credits](https://poly.pizza/l/NoG1sEUD1z/credits): confirma catálogo Kenney, pero se prefirió descarga directa del autor.
- [OpenGameArt / Kenney](https://opengameart.org/content/furniture-kit): espejo del pack CC0; se prefirió fuente Kenney.

## Cerebro

El integrador prepara somas y conexiones reales MaleCNS ligados a replay. Fuente oficial de morfología adicional encontrada: `gs://flyem-male-cns/rois/fullbrain-roi-v4/`, documentada en [MaleCNS download](https://male-cns.janelia.org/download/). La metadata pública tiene 90 regiones, `mesh` legacy y `segment_properties/info`; por ejemplo `mesh/1:0` apunta a `AL(L).ngmesh`. No se descargaron para evitar sustituir la visualización de datos reales elegida por el integrador. Una ROI es volumen/región anatómica, no una neurona ni actividad.

No debe mostrarse actividad ficticia como telemetría, ni etiquetar el modelo de tasas como spikes. Flybody es un recurso anatómico del cuerpo separado del conectoma MaleCNS: no se afirma correspondencia de individuo.

## Comprobación geométrica adicional

Se renderizó el GLB convertido con un visor ortográfico de triángulos independiente (`render_asset.py`), y se inspeccionó `flybody/flybody.qa.png`: la anatomía completa se ensambla correctamente en la pose base. `flybody/flybody.fold.qa.png` muestra un ensayo de articulación de las alas a yaw=1.5 (Z), roll=0.7 (X), pitch=-1 (Y), aplicadas en ese orden local sobre el quaternion base; quedan recogidas hacia atrás y arriba. Es animación, no simulación física ni validación biomecánica. Las imágenes de QA no sustituyen verificación del render web, transparencia y clipping final.

Los GLB Kenney usan Y arriba, y escritorio/silla/monitor miran hacia +Z. La silla tiene el respaldo hacia -Z, asiento hacia +Z. El escritorio tiene cajón hacia +Z. Para que silla y monitor se enfrenten, rotar la silla 180 grados en Y. El asiento está por encima de Y=.19 (esta es la raíz del nodo articulado chair, no la superficie de asiento); usar bounds de vértices y comprobación visual para colocar la mosca. `inspect_glb.py` calcula bounds mundiales leyendo buffers y transformaciones reales de los GLB.
