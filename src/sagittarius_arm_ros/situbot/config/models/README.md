Place local perception model files here on robot deployments.

Recommended YOLO-World file:

- yolov8s-worldv2.pt

Launch example:

roslaunch situbot perception.launch model_weights:=/absolute/path/to/yolov8s-worldv2.pt

For RB8/offline runs, keep allow_model_download:=false so startup fails fast if
the weight file is missing instead of blocking on a network download.
