# DreamMat
DreamMat: High-quality PBR Material Generation with Geometry- and Light-aware Diffusion Models
## [Paper](https://arxiv.org/abs/2405.17176) | [Project page](https://zzzyuqing.github.io/dreammat.github.io/)

![](assets/teaser.png)

### Preparation for inference
1. Install packages in `requirements.txt`.
    We test our model on 3090/4090/V100/A6000 with 11.8 CUDA and 2.0.0 pytorch.
    ```
    git clone https://github.com/zzzyuqing/DreamMat.git
    cd DreamMat
    pip install -r requirements.txt
    ```
2. Install Blender 

    Download [blender-3.2.2-linux-x64.tar.xz](https://download.blender.org/release/Blender3.2/)
    
    Run:
    ```bash
    tar -xvf blender-3.2.2-linux-x64.tar.xz
    export PATH=$PATH:path_to_blender/blender-3.2.2-linux-x64
    ```

3. Download the pre-trained ControlNet checkpoints [here](https://pan.zju.edu.cn/share/78d6588ec65bcfa432ed22d262) or from [hugging face](https://huggingface.co/zzzyuqing/light-geo-controlnet), and put it to the `threestudio_dreammat/model/controlnet`
4. A docker env can be found at https://hub.docker.com/repository/docker/zzzyuqing/dreammat_image/general

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
