# Region Attention Transformer for Medical Image Restoration(RAT)

PyTorch implementation for Region Attention Transformer for Medical Image Restoration (MICCAI 2024).

## Network Architecture

![](README.assets/architecture.JPG)

## Visual Comparison

![](README.assets/vis.JPG)

## Getting Started

- **Mask Prediction & Postprocess**

  First, you can obtain region partitioning masks with the [Segment Anything Model](git@github.com:facebookresearch/segment-anything.git) (SAM) as follows:

  ```python
  from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
  sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
  mask_generator = SamAutomaticMaskGenerator(sam)
  masks = mask_generator.generate(<your_image>)
  ```

  Then, you need to post-process the masks to obtain an indexed mask, which can be then used for compact region partitioning during the downsampling process.

  ```python
  import numpy as np
  def toSegMap(masks): 
      result = np.zeros(masks[0]['segmentation'].shape)
      for i in range(len(masks)): 
          result[masks[i]['segmentation']]=(i+1) 
      result[result==0] = len(masks) + 1
      return result
  masks = sorted(masks, key=itemgetter('area'), reverse=True) 
  indexed_mask = toSegMap(masks)
  ```

- **RAT Inference**

  With the input image and its resultant indexed mask, the output of  RAT can be obtained as follows:

  ```python
  from Model_RAT import RAT
  model=RAT()
  out_img = model(lr_img, indexed_mask) 
  # lr_img shape: [B, C, H, W] 
  # indexed_mask shape: [B, H, W]
  ```

## Citation

If you find AMIR useful in your research, please consider citing:

```bibtex
@misc{yang2024allinone,
      title={All-In-One Medical Image Restoration via Task-Adaptive Routing}, 
      author={Zhiwen Yang and Haowei Chen and Ziniu Qian and Yang Yi and Hui Zhang and Dan Zhao and Bingzheng Wei and Yan Xu},
      year={2024},
      eprint={2405.19769},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

