# YOLO-World + EfficientViT SAM + RAM

## Efficient-SAM URL
https://huggingface.co/mit-han-lab/efficientvit-sam/blob/main/xl1.pt

## :toolbox: Checkpoints
Note : you need to create 'pretrained' folder and download these checkpoints into this folder.
<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Backbone</th>
      <th>Data</th>
      <th>Illustration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>RAM++ (14M)</td>
      <td>Swin-Large</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Provide strong image tagging ability for any category.</td>
      <td><a href="https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth">Download  link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>RAM (14M)</td>
      <td>Swin-Large</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Provide strong image tagging ability for common category.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth">Download  link</a></td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tag2Text (14M)</td>
      <td>Swin-Base</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Support comprehensive captioning and tagging.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
    </tr>
  </tbody>
</table>
