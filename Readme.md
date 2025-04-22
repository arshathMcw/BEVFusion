### Resource
1. Github : [click here](https://github.com/mit-han-lab/bevfusion?tab=readme-ov-file)
1. Paper : [click here](./resources/paper_2205.13542v3.pdf) 
## Prerequisites
1. Point Cloud
### Abstraction
* Multi-sensor fusion is a process that combines data from multiple sensors to create a more complete, accurate, and reliable  understanding of an environment or system .
* Cameras give rich color and texture information (semantic).
* LiDAR gives accurate 3D shape and distance information (geometric).
* BEVFusion is a method for combining data from multiple sensors (like camera and LiDAR) in self-driving systems.
* In old methods, people often project camera features into the LiDAR point cloud, and this process loses a lot of useful camera details (like meaning or context of objects). That makes it hard to do tasks like segmenting a 3D scene into cars, roads, trees, etc.
* BEVFusion solves this by combining all sensor data into a Bird’s-Eye View (BEV) format, which keeps both geometric (LiDAR) and semantic (camera) information.
* It also improves speed using an optimized BEV pooling, making it over 40× faster.

### Introduction
* Self-driving cars use many types of sensors. For example, Waymo cars use: (29 cameras,6 radars and 5 LiDARs.)
* Since each sensor gives different types of information, combining their data (multi-sensor fusion) is very important for accurate understanding.But each sensor gives data in a different format:
    * Cameras → 2D perspective view
    * LiDAR → 3D point cloud
* To combine them, we need a common format that works for all sensors and tasks.One idea is to project the 3D LiDAR data onto the 2D camera view and use 2D CNNs. But doing this distorts the 3D geometry, which makes it less effective for tasks like 3D object recognition.
* Earlier methods tried to project LiDAR data into the camera view, but that caused distortion.
* So, recent methods try the opposite — they keep the LiDAR point cloud as it is, and instead add extra information from the camera to it,Here’s how they do that:
    * Semantic labels: Add tags like "car", "person", or "road" to each LiDAR point using the camera.
    * CNN features: Take deep visual details (from camera images processed by a CNN) and attach them to LiDAR points.
    * Virtual points: Create fake (virtual) 3D points using images to add more detail to the point cloud.
* After doing this, they just use a LiDAR model to analyze the enhanced point cloud and make predictions.And this methods works for Object detection but have lossy segmenatation 
* But they don’t work well for semantic tasks like BEV map segmentation (e.g., labeling lanes, sidewalks, cars on a top-down map).
* Why? Because when projecting camera features to LiDAR points, most features get dropped:
* A 32-beam LiDAR can only match ~5% of camera features.
* The rest (95%) are lost, causing a semantic loss (see Figure 1b).
* This problem is worse with sparser sensors (like low-resolution LiDAR or radar).
* BEVFusion is a new method that combines data from different sensors (like LiDAR and cameras).
* It does this by converting everything into a Bird’s-Eye View (BEV) — a top-down view of the scene.
* BEVFusion combines data from different sensors (like cameras and LiDAR) into a bird's-eye view (BEV). This helps the system understand the environment better by looking at everything from above in a unified way.
* It keeps important details about both geometry (shape and space) and semantics (meaning or labels) when converting data into this BEV format.
* This approach is useful for various tasks that deal with 3D perception, like detecting objects or tracking them, because these tasks can be naturally handled in the BEV representation.
* However, during the process, there's a speed issue with how the data is transformed into BEV, especially when performing a pooling operation. This step takes up most of the time (over 80%).
* To solve this, a faster method is introduced using a specialized kernel with tricks like precomputing some calculations in advance, which results in a 40× speedup.
* Finally, the system uses a convolutional network to combine features from all sensors and can be adapted to different tasks by adding specific layers on top.
* BEVFusion sets a new record in 3D object detection performance on nuScenes and Waymo benchmarks, outperforming all other methods, even those using advanced techniques like test-time augmentation and model ensemble.
* It shows a significant boost in BEV map segmentation, achieving 6% higher mIoU than camera-only models and 13.6% higher mIoU than LiDAR-only models, while other fusion methods fail in this task.
* On top of that, BEVFusion is more efficient, delivering these results with 1.9× lower computation cost.
For the past few years, point-level fusion has been the dominant approach, but BEVFusion challenges this by questioning whether LiDAR space is the best place for sensor fusion.
* It introduces a new perspective by using a bird’s-eye view (BEV) representation, which has proven to be more effective, offering superior performance over traditional methods.
* The simplicity of BEVFusion is also a key strength, making it easy to adopt and extend.
* The paper hopes that this work will establish BEVFusion as a strong baseline for future sensor fusion research and encourage researchers to rethink the design and approach for multi-task, multi-sensor fusion.

### Related Work
1. LiDAR-Based 3D Perception:
    * Several researchers have developed single-stage 3D object detectors that flatten LiDAR point cloud features using methods like PointNets or SparseConvNet to perform detection in the BEV space.
    * Others have explored anchor-free single-stage 3D detection, as well as two-stage object detector designs, adding an RCNN network to enhance the performance of one-stage detectors.
1. Camera-Based 3D Perception:
    * Due to the high cost of LiDAR, many researchers focus on camera-only 3D perception. One approach, FCOS3D, extends image detectors with 3D regression branches, and other studies have improved depth modeling.
    * Instead of detection in perspective view, some methods convert camera features into BEV using a view transformer, which has been explored in models like BEVDet and M2BEV.
    * Research on multi-head attention has also been applied to view transformation for 3D object detection.
1. Multi-Sensor Fusion:
    * Multi-sensor fusion has become a prominent area in 3D detection. Approaches can be categorized into proposal-level and point-level fusion:
    * Proposal-level fusion focuses on creating object proposals in 3D and fusing image features with them. Examples include MV3D, FUTR3D, and TransFusion.
    * Point-level fusion typically involves mapping image semantic features onto LiDAR points to perform detection.
    * BEVFusion, however, performs fusion in a shared BEV space, treating both foreground/background and geometric/semantic information equally, offering a more generalized framework for multi-task, multi-sensor perception.
### Methods
BEVFusion focuses on fusing data from multiple sensors—like LiDAR and multiple cameras—to handle multiple 3D perception tasks such as object detection and scene segmentation. It starts by using separate encoders for each type of input to extract their unique features. These features are then transformed into a shared BEV format that holds both geometric structure and semantic meaning. Since this transformation process is computationally heavy, it gets optimized through precomputation and interval reduction, making it much faster. After that, a convolution-based encoder is applied to refine the unified BEV features and correct any local misalignments between inputs. The final step adds task-specific components to perform different 3D perception tasks.
#### Unified Representation
* The challenge of multi-sensor fusion stems from the differences in the features captured by various sensors, such as cameras and LiDAR. Each sensor captures information from different viewpoints, making it difficult to align their features directly. Here's a breakdown of the challenges and why using the Bird's-Eye View (BEV) as a unified representation solves many of these issues:
1. View Discrepancy: Different sensors provide data in distinct views:
    * Camera features are captured in the perspective view, where the object or scene is represented with varying depths and distortions.
    * LiDAR features are usually captured in the 3D/bird’s-eye view, representing the environment in a top-down manner, preserving depth and geometric structure.
    * This discrepancy leads to spatial misalignment when trying to fuse these features directly, making it challenging to apply simple elementwise fusion.
1. Geometrically Lossy Projections:
    * Camera to LiDAR: Projecting LiDAR points to a camera plane leads to a 2.5D depth map. This process is geometrically lossy, meaning two adjacent points in the depth map could be far apart in 3D space, which reduces the camera view's utility for tasks requiring geometric precision like 3D object detection.
    * LiDAR to Camera: When camera features are projected onto LiDAR points, only a small fraction of the camera’s features can be matched to the LiDAR points. This is due to the drastically different densities between camera and LiDAR features, resulting in less than 5% of camera features matching LiDAR points (in the case of a 32-channel LiDAR scanner). This loss of semantic density diminishes the performance on tasks requiring detailed semantic information, such as BEV map segmentation.
1. The Advantage of BEV: The Bird's-Eye View (BEV) provides a unified representation that avoids the issues of geometrically and semantically lossy projections:
    * LiDAR to BEV: The transformation of LiDAR features into the BEV space flattens the sparse 3D LiDAR points along the height dimension, preserving the geometric structure without distorting the environment.
    * Camera to BEV: The transformation of camera features into BEV involves projecting each camera pixel back into a ray in the 3D space. This preserves the dense semantic information from the camera while providing a coherent, top-down view of the environment. The BEV map that results from this fusion retains both geometric structure (from LiDAR) and semantic density (from cameras), which is crucial for tasks like 3D object detection and BEV map segmentation.

#### Efficient camera to BEV Transformation
The process of transforming camera data into Bird's Eye View (BEV) representation is a crucial step in 3D detection tasks, particularly for systems that integrate multi-modal sensory data. However, this transformation is computationally expensive and faces challenges in efficiency, especially when dealing with dense camera feature point clouds.

Here’s an explanation of the methods used to optimize this transformation:

1. Depth Distribution and Camera Feature Point Cloud
The key challenge in camera-to-BEV transformation is the depth ambiguity for each pixel in the camera feature map. To address this:
    Each pixel’s depth distribution is explicitly predicted, rather than assuming a single depth value.
    The camera feature pixels are scattered along D discrete points along the camera ray, each weighted by their respective depth probability.
    This results in a 3D camera feature point cloud of size 𝑁×𝐻×𝑊×𝐷
    N: Number of cameras.H, W: Camera feature map size.D: Number of discrete depth values.
    The feature point cloud is quantized along the x, y axes with a specified step size (e.g., 0.4m), and the BEV pooling operation aggregates the features within each r × r BEV grid, flattening the features along the z-axis.

2. Challenges with Existing BEV Pooling
While the BEV pooling operation is conceptually simple, it is surprisingly inefficient and slow, taking over 500ms on an RTX 3090 GPU for a typical workload. This inefficiency arises because the camera feature point cloud is large and dense, with up to 2 million points generated for each frame, which is much denser than a LiDAR feature point cloud.

3. Optimizing BEV Pooling with Precomputation and Interval Reduction
To improve the efficiency of BEV pooling, two key optimizations are introduced: Precomputation and Interval Reduction.

##### Precomputation
    * The first step of BEV pooling is to associate each point in the camera feature point cloud with a corresponding BEV grid.

    * Unlike LiDAR point clouds, the camera feature points have fixed coordinates, as long as the camera intrinsics and extrinsics are constant (after calibration).

    * The precomputation step involves:

        * Precomputing the 3D coordinates and BEV grid index for each point.

        * Sorting the points according to their grid indices and recording the rank of each point.

        * During inference, the feature points are reordered based on precomputed ranks, reducing the latency of grid association from 17ms to 4ms.

##### Interval Reduction
    * After grid association, the next step is feature aggregation within each BEV grid using symmetric functions (e.g., mean, max, or sum).
    * Traditional methods, like the prefix sum, compute cumulative sums across all points, then subtract boundary values where indices change. This process requires tree reduction on the GPU, leading to inefficiencies due to unused partial sums.
    * Interval Reduction optimizes this by introducing a custom GPU kernel:
    * Each GPU thread is assigned to a specific BEV grid to calculate its interval sum and write the result directly.
    * This avoids multi-level tree reductions and unnecessary memory accesses, improving speed.
    * As a result, the feature aggregation latency is reduced from 500ms to 2ms.

4. Performance Improvements and Takeaways
    * With these optimizations, camera-to-BEV transformation becomes significantly faster:
    * The overall latency is reduced from 500ms to 12ms (a 40x speedup), making it only 10% of the total model runtime.
    * The optimized approach scales well with different feature resolutions, making it adaptable to various input sizes and workloads.
    * Compared to concurrent methods that approximate the view transformer (by assuming uniform depth distributions or truncating points within BEV grids), this approach is exact and still achieves faster processing times.

#### Fully-Convolutional Fusion in BEV Representation
Once all the sensory features (e.g., from LiDAR and cameras) are converted to a shared Bird's Eye View (BEV) representation, it becomes straightforward to fuse them together using elementwise operations, such as concatenation. However, there can still be spatial misalignments between the LiDAR and camera BEV features due to inaccuracies in depth estimation from the view transformer.

1. Addressing Misalignments: Convolution-Based BEV Encoder
    * To handle these misalignments, a convolution-based BEV encoder is employed. This encoder helps to align the features from different modalities (LiDAR and camera) by applying convolutional layers (sometimes with residual blocks). The convolutional layers allow for spatial corrections and adjustments, aligning the features from different sensory sources in the BEV space.
    * The encoder smooths out the misalignments that might be introduced during the depth transformation, making the multi-modal features more compatible.
    * The convolutional layers are typically designed to capture local patterns and spatial relations, which helps in compensating for the misaligned features between LiDAR and camera inputs.

2. Potential for Improvement with Accurate Depth Estimation
    * The current method relies on the existing depth estimation from the view transformer, but it could benefit from more accurate depth estimation. For instance:
    * Supervising the view transformer with groundtruth depth could improve the depth accuracy, which would, in turn, enhance the alignment of the sensory features in the BEV representation.
    * This potential improvement is highlighted as future work to further refine the model.

#### Multi-Task Heads for 3D Perception Tasks
After fusion, task-specific heads are applied to the fused BEV feature map to handle various 3D perception tasks. These tasks can include:
1. 3D Object Detection
    For 3D object detection, the method utilizes:
        * Class-specific center heatmap heads to predict the center locations of all objects.
        * Regression heads for estimating other properties of the objects, such as size, rotation, and velocity.
1. Map Segmentation
    * For map segmentation tasks, where categories may overlap (e.g., crosswalks as part of drivable spaces), the segmentation task is modeled as multiple binary semantic segmentation tasks, one for each class:
    * Focal loss is used to train the segmentation heads, following the approach of CVT [8].
    * By using multiple task-specific heads, the method is highly versatile and can address a range of different perception problems effectively.

### Experiments
#### Model
The model uses two types of backbones to process different types of data: a Swin Transformer (Swin-T) for camera images and VoxelNet for LiDAR point clouds. Swin-T is a vision transformer that efficiently extracts features from images by analyzing them in a hierarchical manner, while VoxelNet converts the 3D LiDAR point cloud into a voxel grid (small cubes in 3D space) and applies 3D convolutions to extract meaningful features. To ensure that features at different scales (both small and large objects) are effectively captured, the model uses a Feature Pyramid Network (FPN), which fuses multi-scale features from the image backbone. Before feeding images into the network, they are resized to 256×704 to reduce computational load. The LiDAR data is voxelized using different resolutions: 0.075 meters for detection tasks (which need more precision) and 0.1 meters for segmentation tasks (which can be slightly coarser). Since detection and segmentation require different bird’s-eye view (BEV) feature map sizes and ranges, the model uses a technique called grid sampling with bilinear interpolation to resize and align these feature maps accordingly. This allows the model to share a common BEV representation while still adapting it to different tasks. Finally, task-specific heads are applied to the processed BEV features—for example, one for detecting objects and another for segmenting map areas.
### Dataset
The model is evaluated on two popular and large-scale datasets for 3D perception: nuScenes and Waymo. Both of these datasets contain over 40,000 annotated scenes, meaning each scene has labeled information like where cars, pedestrians, and other objects are located. What's special about these datasets is that each sample (scene) includes data from both LiDAR sensors and multiple surrounding cameras. This allows the model to learn from and make predictions based on data from different sensor types—LiDAR provides accurate depth, while cameras provide rich visual details. These datasets are widely used for testing autonomous driving systems and 3D perception models.
