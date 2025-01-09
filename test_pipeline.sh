gst-launch-1.0 \
    filesrc location=test_video.mp4 ! decodebin ! videoconvert ! \
    gvadetect model=yolov8n.xml model-proc=yolov8n.json device=CPU inference-interval=1 ! queue ! \
    gvaclassify model=reid_model.xml model-proc=reid_model.json device=CPU inference-interval=1 ! queue ! \
    gvawatermark ! videoconvert ! autovideosink
