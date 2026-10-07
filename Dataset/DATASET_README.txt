SPORTS AISD DATASET

Total Images
------------
916

Classes
-------

Basketball Shoot
Cricket Batting
Tennis Swing
Volleyball Spike

Image Size
----------

640 × 640 pixels

Annotation Tool
---------------

LabelImg

Annotation Format
-----------------

YOLO TXT

Dataset Sources
---------------

1. UCF101 Sports Action Dataset
2. Self-recorded cricket batting video

Dataset Preparation
-------------------

- Videos converted into image frames.
- Images manually annotated using LabelImg.
- Class IDs remapped from the original 6-class format to the final 4-class format.
- Duplicate, blurred, and incorrect images removed.
- Images split into training, validation, and testing sets.

Dataset Split
-------------

Training      80%
Validation    10%
Testing       10%
