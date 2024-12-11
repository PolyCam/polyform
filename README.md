# polyform

Polycam is an application for 3D scanning that can be downloaded for free on the App Store [here](https://apps.apple.com/us/app/polycam-lidar-3d-scanner/id1532482376). While the main purpose of Polycam is to reconstruct 3D meshes, along the way we collect and optimize images, camera intrinsic and extrinsic data as well as depth maps. We have heard from numerous members of the community that this raw data could be useful for them, for applications such as training NeRFs, improving indoor scene reconstruction or doing more analysis than what can be done in Polycam alone. 

This project, polyform, includes tools for converting from Polycam's raw data format into other data formats, such as those used to train NeRFs. As discussed in more detail below, the camera poses are globally optimized and do not require any further optimization (i.e. COLMAP is not required) to be used as input to a NeRF.

We hope that by exposing a free raw data export option, as well as documentation and tools for working with this data, we can help to enable researchers working in the field of 3D computer vision, as well as creative technologists working on the forefront of 3D creation. If you make something cool and post it online please tag us @polycam3d.

## Requirements & preliminary remarks

### An iOS device with LiDAR is required

Raw data with images, camera poses and depth maps are only available when running Polycam on an iOS device with LiDAR (i.e. 2020+ iPad Pro, or iPhone 12+ Pro), and only available for sessions captured in LiDAR mode or ROOM mode.

### Developer mode must be turned on

Open the settings screen in the Polycam iOS app, scroll down to 'Developer mode' and turn it on. This will expose a 'raw data' export option in the data export menu for all lidar captures. 

**NOTE:** Because developer mode turns on data caching of intermediate artifacts, it does not apply retroactively. You will need to either re-process existing datasets or else collect new datasets to generate the full raw data needed for export.

### Installation and setup

Python 3.7+ is recommended.

```
pip3 install numpy fire Pillow # install dependencies
git clone https://github.com/PolyCam/polyform
```

### Polycam poses are globally optimized

Unlike most other 3D scanning apps, we globally optimize the ARKit camera poses before reconstructing the mesh, and these optimized camera poses should be as good or better than what you would get from an SfM software like COLMAP, particularly for complex indoor scenes where most SfM pipelines will fail. This means you do not need to worry about drift while scanning, and can loop back and scan the same area multiple times in a single session.

Given that the pose optimization can take some time to run on very lage scenes, there are a few limitations: (i) by default we only run pose optimization on scenes with less than 700 frames but (ii) if you open up the CUSTOM processing panel you can turn on pose optimization (aka Loop Closure) for scenes up to 1400 frames. If the 'Loop Closure' option is disabled that means the scene is too large to optimize on device, but you can still export the ARKit poses.

## Converting from Polycam's data format to other formats


### Nerfstudio

[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) offers a suite of tools for training, visualizing and working with NeRFS. They offer a tool to directly import Polycam's data format, and we recommend using their importer if you wish to use nerfstudio. See [their docs](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html#polycam-capture) for more info.

### Instant-ngp

[Instant-NGP](https://github.com/NVlabs/instant-ngp) is a popular open source project for training and visualizing NeRFs. You can convert from Polycam's data format to the the instant-ngp format by running the following command from the root of the `polyform` repo:

```
python3 -m polyform.convert <path-to-data-folder> --format ingp
```

Note: you may need to tweak the `scale` parameter in the transforms.json file to get the best results. 

### Adding additional convertors:

If you would like to add an additional export format you can do so by consulting Polycam's data specification below, and using `polyform/convertors/instant_ngp.py` as an example.

## Polycam's data format

The raw data exported from Polycam has the following form:

```
capture-folder/
    raw.glb
    thumbnail.jpg
    polycam.mp4
    mesh_info.json
    keyframes/
        images/
        corrected_images/
        cameras/
        corrected_cameras/
        depth/
        confidence/
```

we will describe each of these artifacts below.

### raw.glb, thumbnail, video, mesh_info

The `raw.glb` file contains the reconstructed mesh in binary gLTF form. This is put into an axis-aligned coordinate system, and if you wish to put it into the same coorindate system as the raw data you will need to apply the inverse of the `alignmentTransform` contained in the `mesh_info.json` (stored in column-major format).

The `thumbnail.jpg` and `polycam.mp4` are visual metadata which are included for convenience, and the `mesh_info.json` contains some metadata about the raw.glb mesh such as the number of faces and vertices.

### images

This directory contains the original images, where the name of the file is the timestamp in microseconds, and will match the filenames for the other corresponding files (camera, depth map etc). Note that these images have some distortion, so you will probably want to use the `corrected_images` instead.

### corrected_images

This directory contains the undistorted images that are computed during the pose optimization step. For best results you will want to use these images, although you may need to mask out the black pixels around the border that are a result of the undistortion process. If this directory does not exist, it means the session was too large for the pose optimization step to run on device.

### cameras

This directory includes the `camera.json` files that hold the camera intrinsics and extrinsics, as well as the image `width`, `height` and a `blur_score` for the image, where a lower score means more blur.

The `t_ij` element of the transform matrix is in row-major order, so `i` is the row and `j` is the column. We omit the last row becuase it is simply `[0, 0, 0, 1]`.

**NOTE:** The coordinate system of the camera poses follows the ARKit gravity aligned convention which you can read about [here](https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/gravity). In short, this is a right-handed coordinate system where the `+Y axis` points up, and `-Z axis` points in the direction that the camera was pointing when the session started, and perpendicular to gravity. Note: depending on the coordinate system convention of your downstream application you might have to apply an additional rotation to the camera transform to get the conventions to match.

### corrected_cameras

This directory includes the improved camera poses that have been globally optimized. The data is formatted in the same way as the cameras described above, and you should almost always use these files intead of the original camera files. If this directory does not exist, it means the session was too large for the pose optimization step to run on device, but you can still fall use the unoptimized ARKit camera poses.

### depth

This directory contains the depth maps acquired during the scanning session (see [docs](https://developer.apple.com/documentation/arkit/arframe/3566299-scenedepth)). These depth maps are generated by ARKit using the LiDAR sensor. A couple things to note:

- Because the resolution is lower than the image resolution, you will have to scale the camera intrinsics to match the depth map resolution if you wish to use them.
- The depth maps are not perfect -- because the lidar sensor is low resolution, and the ML-based upresolution process can introduce artifacts, they may not be useful for some applications. In general, resolving geometric detail less than 1-2 cm is not possible, and sometimes they can contain gross errors.
- The max range of the lidar sensor is 5M, so the depth map for any part of the scene larger than that is a total guess.

That being said, they are pretty good for indoor scenes or other scenes that do not require fine geometric detail.

The depth data is stored in lossless .png format as a single channel image where depth is encoded as a 16-bit integer with units of millimeters, so a value of 1250 would correspond to 1.25 meters.

### confidence

This folder includes the pixel-wise confidence maps associated with the depth maps (see [docs](https://developer.apple.com/documentation/arkit/ardepthdata/3566295-confidencemap)). ARKit only provides three confidence values -- low, medium and high. We encode these as a single-channel 8-bit integer map, where 0 = low, 127 = medium, and 255 = high. The low confidence values are quite unreliable, and for most applications it would be best to filter them out.


## Tips on dataset collection

 To get the best datasets, for training NeRFS or just in general, it's important to (i) get great coverage of your subject and (ii) reduce blur as much as possible. To reduce blur it's important to have good lighting and move the camera slowly if using auto mode, but to get the clearest images, particularly for lower light conditions such as indoors, it's recommended to use the manual shutter mode to collect still images -- you will just need to be sure to collect a lot of them.

 ## Working at Polycam

 If you are an engineer or researcher working in the field of 3D computer vision and want to work on a product that's shipping 3D reconstruction to millions of users, please visit: https://poly.cam/careers.
