import argparse
import numpy as np
import trimesh


if __name__ == "__main__":

    # Script arguments
    parser = argparse.ArgumentParser(description='View Object Stable Poses and Grasps')
    parser.add_argument('obj_data_path', type=str, help='path to object data file')
    args = parser.parse_args()


    # Extract data for mesh
    obj_data = np.load(args.obj_data_path)
    mesh = trimesh.Trimesh(vertices=obj_data["mesh_vertices"], 
                           faces=obj_data["mesh_faces"])
    
    stps = obj_data["stps"]
    stp_probs = obj_data["probs"]

    for i, pose, prob in zip(range(len(stps)), stps, stp_probs):
        print('Stable Pose {:d} (lambda_{:d} = {:.4f})'.format(i,i,prob))
        mesh.apply_transform(pose)
        mesh.show()