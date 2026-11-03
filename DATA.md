# Data access and sample identity

Official CitrusFarm source: https://ucr-robotics.github.io/Citrus-Farm-Dataset/

Original publication: Hanzhe Teng, Yipeng Wang, Xiaoao Song and Konstantinos Karydis. Multimodal Dataset for Localization, Mapping and Crop Monitoring in Citrus Tree Farms. ISVC 2023, pp. 571-582. https://arxiv.org/abs/2309.15332

The original provider supplies ROS bags and extraction tools. The study uses registered stereo RGB/depth pairs. Each manifest retains the recording-file basename, RGB and depth timestamps, sample identifier and split. Absolute laboratory paths have been replaced with relative paths. Collection files can be joined by recording basename and timestamps; matching an image filename alone is not a substitute for checking the paired depth identity.

Prepared file layout:

```text
prepared_citrusfarm/
  images_v2/<sample_id>.png
  depths_v2/<sample_id>.npy
```

RGB files retain their original 1280 by 720 dimensions. Depth NPY arrays contain metric reference depths in metres. The reader converts these to float32 and uses nearest-neighbour sampling on the 518 by 518 evaluation grid. `sare_depth/data.py` checks path containment, file existence and duplicate sample identifiers. The evaluator checks the reference range and invalid predictions.

The complete study-specific extraction ledger from each original bag message to every prepared file is not distributed. The manifests preserve the member identity and recorded timestamps, while the two included examples make the inference/evaluation entry point immediately executable. For full-set execution, place the prepared image/depth pairs at the manifest paths after acquiring the corresponding source files from the provider.

The `example_data` pairs were extracted from CitrusFarm. PNG/NPY are packaging conversions of the source RGB/registered depth, not synthetic imagery. Attribution and CC BY-SA 4.0 apply to these extracted examples: https://creativecommons.org/licenses/by-sa/4.0/

The private Dingxi field images and measurement records are not included in this release. The manuscript describes their separate object-distance comparison.
