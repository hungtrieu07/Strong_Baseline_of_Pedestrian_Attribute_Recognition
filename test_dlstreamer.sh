# gst-launch-1.0 filesrc location=vehicle.mp4 ! decodebin ! videoconvert ! \
#                 gvadetect model=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/best.xml model-proc=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/model-proc.json device=CPU inference-interval=1 ! queue ! \
#                 gvaclassify model=openvino_models/reid/reid_model.xml model-proc=openvino_models/reid/reid_model.json device=CPU inference-interval=1 ! queue ! \
#                 gvawatermark ! videoconvert ! autovideosink

# gst-launch-1.0 filesrc location=vehicle.mp4 ! decodebin ! videoconvert ! \
#                 gvadetect model=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/best.xml model-proc=openvino_models/yolov9-retrained-human-vehicle/best_int8_openvino_model/model-proc.json device=CPU inference-interval=1 ! queue ! \
#                 gvawatermark ! videoconvert ! autovideosink

gst-launch-1.0 filesrc location=vehicle.mp4 ! decodebin ! videoconvert ! \
                gvadetect model=openvino_models/yolov9-retrained-human-vehicle/best.xml model-proc=openvino_models/yolov9-retrained-human-vehicle/model-proc.json device=CPU inference-interval=5 ! queue ! \
                gvaclassify model=openvino_models/reid/reid_model.xml model-proc=openvino_models/reid/reid_model.json device=CPU inference-interval=5 ! queue ! \
                gvawatermark ! gvametaconvert format=json ! \
                videoconvert ! autovideosink sync=false