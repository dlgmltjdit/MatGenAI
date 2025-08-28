# MatGenAI
We propose MatGenAI, a multi-pipeline DreamMat framework that integrates 3D mesh semantic segmentation (Sampart3D) with a Large Language Model (LLM). Our approach fix up the mismatch between semantic descriptions (text prompts) and local features.

### Limitation of dreammat
![](assets/example.png)

## Method
### Proposed pipeline
![](assets/pipeline_matgenai.png)

### (a) Extended pipeline
1. The input 3D mesh is segmented into sub-meshes based on semantic units.
2. The semantic information of each sub-mesh is extracted in textual form using an LLM.
3. It is extended into a sub-pipeline conditioned on (sub-mesh, semantic description).

### (b) Training step
1. Each pipeline independently predicts PBR materials by following the material prediction methodology of DreamMat.
2. Each PBR material — Albedo, Metallic, and Roughness — is defined according to the hash-grid function.

### (c) Texture export
1. Each sub-mesh is normalized to a global coordinate system centered at the origin (0,0,0) by an affine transformation matrix.
2. Therefore, a coordinate transformation process is performed to apply the generated sub-textures back to the original mesh .

### results
![](assets/results.png)

# How to use
You must follow the Installation Guide on each project's GitHub page.
1. Install DreamMat for inference PBR Material
[DreamMat](https://zzzyuqing.github.io/dreammat.github.io/)

3. Prepare SAMPart3D for semantic segmentation
[SAMPart3D](https://github.com/Pointcept/SAMPart3D)

5. To input your 3D mesh file (ex, knight.obj) into SAMPart3D and get a .npy file containing the per-part segmentation information as output.
6. Instead of the knight example, use your own filenames and place them according to the structure below.
```
threestudio_dreammat/load
|-- shapes
    |-- objs
        |-- knight.obj
    |-- seg
        |-- knight.npy
```
6. Run MatGenAI.
```
cd MatGenAI/MatGenAI/threestudio_dreammat
sh cmd/run_examples.sh
```
