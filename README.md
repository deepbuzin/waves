# waves
This is a test assignment for a computer vision position.

### Description
rep_counter.py contains the very bare-bones algorithm that calculates the distance between the right shoulder and the right wrist, and compares it to terminal points' values.
Normalization, centering and other possible preprocessing steps were omitted here, since neither of provided videos call for any generalization capacity.

thresh_estimator.py is a helper script that collects the distance data from the video and calculates their local extrema, that were used to set terminal point thresholds.