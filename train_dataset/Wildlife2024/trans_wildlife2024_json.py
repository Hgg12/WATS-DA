import os
import json


if __name__ == "__main__":
    root = "tracker/BAN/train_dataset/WATB2024_1/result/annotations"
    target = 'tracker/BAN/train_dataset/WATB2024_1/result/sam_WATB2024_1.json'

    json_list = os.listdir(root)
    _json = {}
    for json_file_name in json_list:
        json_file = json.load(open(os.path.join(root, json_file_name)))
        for video_name, frames in json_file.items():
            if video_name == "0000box1":
                print()

            _frames = {}
            for frame in frames:
                for frame_name, boxes in frame.items():
                    _boxes = {}
                    for box_idx, box in enumerate(boxes):
                        _boxes["%06d" % box_idx] = box
                _frames[frame_name] = _boxes
            _json[video_name] = _frames
    json.dump(_json, open(target, 'w'), indent=4, sort_keys=True)
