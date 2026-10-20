string action # 現状は take_out, bring_in
BoundingBox bounding_box
sensor_msgs/CompressedImage mask
string class_id # YOLOのクラスラベル (例: book)。bg_subtraction系パイプラインでは空文字
