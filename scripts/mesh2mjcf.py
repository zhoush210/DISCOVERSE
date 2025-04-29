import os
import shutil
import argparse
from discoverse import DISCOVERSE_ASSETS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="The path to the mesh file (.obj or .stl), use full path.")
    parser.add_argument("--rgba", nargs=4, type=float, default=[0.5, 0.5, 0.5, 1], help="The rgba color of the mesh, default is [0.5, 0.5, 0.5, 1].")
    parser.add_argument("--texture", type=str, default=None, help="The path to the texture file (.png), use full path.")
    parser.add_argument("--mass", type=float, default=0.001, help="The mass of the mesh, default is 0.001.")
    parser.add_argument("--diaginertia", nargs=3, type=float, default=[0.00002, 0.00002, 0.00002], help="The diagonal inertia of the mesh, default is [0.00002, 0.00002, 0.00002]." )
    parser.add_argument("--fix_joint", action="store_true", help="Whether to fix the joint of the mesh, default is False.")
    parser.add_argument("-cd", "--convex_decomposition", action="store_true", help="Whether to decompose the mesh into convex parts, default is False.")
    parser.add_argument("--verbose", action="store_true", help="Whether to visualize the mesh, default is False.")
    args = parser.parse_args()

    verbose = args.verbose
    convex_de = args.convex_decomposition
    if convex_de:
        try:
            import coacd
            import trimesh
        except ImportError:
            print("Error: coacd and trimesh are not installed. Please install them using 'pip install coacd trimesh'.")
            exit(1)

    input_file = args.input_file
    rgba = args.rgba
    mass = args.mass
    diaginertia = args.diaginertia
    fix_joint = args.fix_joint

    if input_file.endswith(".obj"):
        asset_name = input_file.split("/")[-1].replace(".obj", "")
    elif input_file.endswith(".stl"):
        asset_name = input_file.split("/")[-1].replace(".stl", "")
    else:
        exit(f"Error: {input_file} is not a valid file type. Please use .obj or .stl files.")

    if args.texture is not None and os.path.exists(args.texture):
        texture_name = os.path.basename(args.texture)
        texture_dir = os.path.join("obj", asset_name, texture_name)
        texture_target_path = os.path.join(DISCOVERSE_ASSETS_DIR, "textures", texture_dir)
        if not os.path.exists(os.path.dirname(texture_target_path)):
            os.makedirs(os.path.dirname(texture_target_path))
        shutil.copy(args.texture, texture_target_path)
        rgba[3] = 0.0
    else:
        texture_name = None
    
    output_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes", "obj", asset_name)    
    mjcf_obj_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "object")
    if not os.path.exists(mjcf_obj_dir):
        os.makedirs(mjcf_obj_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if os.path.dirname(input_file) != output_dir:
        shutil.copy(input_file, output_dir)
    asset_config = """<mujocoinclude>\n  <asset>\n"""

    geom_config = f"""<mujocoinclude>\n"""
    if not fix_joint:
        geom_config += f'  <joint type="free"/>\n'
    geom_config += f'  <inertial pos="0 0 0" mass="{mass}" diaginertia="{diaginertia[0]} {diaginertia[1]} {diaginertia[2]}" />\n'

    if texture_name is not None:
        asset_config += f'    <texture type="2d" name="{asset_name}_texture" file="{texture_dir}"/>\n'
        asset_config += f'    <material name="{asset_name}_texture" texture="{asset_name}_texture"/>\n\n'
        geom_config += f'  <geom material="{asset_name}_texture" mesh="{asset_name}" class="obj_visual"/>\n\n'

    asset_config += '    <mesh name="{}" file="obj/{}/{}.obj"/>\n'.format(asset_name, asset_name, asset_name)

    if convex_de:
        mesh = trimesh.load(input_file, force="mesh")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh)

        for i, part in enumerate(parts):
            output_file = os.path.join(output_dir, f"part_{i}.obj")
            part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
            part_mesh.export(output_file)
            asset_config += '    <mesh name="{}_part_{}" file="obj/{}/part_{}.obj"/>\n'.format(asset_name, i, asset_name, i)
            geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}_part_{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name, i)
        print(f"Asset {asset_name} has been decomposed into {len(parts)} parts")
    else:
        geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name)

    asset_config += '  </asset>\n</mujocoinclude>\n'
    asset_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}_dependencies.xml")
    with open(asset_file_path, "w") as f:
        f.write(asset_config)

    geom_config += '</mujocoinclude>\n'
    geom_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}.xml")
    with open(geom_file_path, "w") as f:
        f.write(geom_config)

    print(f"Asset {asset_name} has been converted to mjcf and saved to {output_dir}")
    print(f"Asset file: {asset_file_path}")
    print(f"Geom file: {geom_file_path}")

    if verbose:
        py_dir = os.popen('which python3').read().strip()
        tmp_world_mjcf = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "_tmp.xml")

        tmp_content = ""
        tmp_content += '<mujoco model="temp_env">\n'
        tmp_content += '  <option gravity="0 0 0"/>\n'
        tmp_content += '  <compiler meshdir="../meshes" texturedir="../textures/"/>\n'
        tmp_content +=f'  <include file="object/{asset_name}_dependencies.xml"/>\n'
        tmp_content += '  <default>\n'
        tmp_content += '    <default class="obj_visual">\n'
        tmp_content += '      <geom group="2" type="mesh" contype="0" conaffinity="0"/>\n'
        tmp_content += '    </default>\n'
        tmp_content += '  </default>\n'
        tmp_content += '  <worldbody>\n'
        tmp_content +=f'    <body name="{asset_name}">\n'
        tmp_content +=f'      <include file="object/{asset_name}.xml"/>\n'
        tmp_content += '    </body>\n'
        tmp_content += '  </worldbody>\n'
        tmp_content += '</mujoco>\n'

        with open(tmp_world_mjcf, "w") as f:
            f.write(tmp_content)
        cmd_line = f"{py_dir} -m mujoco.viewer --mjcf={tmp_world_mjcf}"
        print(cmd_line)
        os.system(cmd_line)
        os.remove(tmp_world_mjcf)