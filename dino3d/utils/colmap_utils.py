import struct
import numpy as np
import os

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = {
                0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 4,
                7: 5, 8: 8, 9: 12
            }[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "model": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read image name as null-terminated string
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_bytes.append(c)
            image_name = b''.join(name_bytes).decode('utf-8')

            # Read 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2D):
                x, y, p3d = struct.unpack("<ddq", f.read(24))
                xys.append((x, y))
                point3D_ids.append(p3d)

            point3D_ids = np.array(point3D_ids, dtype=np.int64)
            mask = point3D_ids >= 0 # fiilter -1 values
            point3D_ids = point3D_ids[mask]
            if point3D_ids.size and point3D_ids.max() > np.iinfo(np.int32).max:
                raise ValueError("Some point3D_id exceed int32 range.")
            point3D_ids = point3D_ids.astype(np.int32)

            xys = np.array(xys, dtype=np.float32)
            xys = xys[mask]

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images