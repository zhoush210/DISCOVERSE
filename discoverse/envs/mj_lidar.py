import mujoco
import numpy as np

class MjLidarSensor:
    mjtGeom2Func = {
        0 : "ray_intersect_PLANE",
        2 : "ray_intersect_SPHERE",
        3 : "ray_intersect_CAPSULE",
        4 : "ray_intersect_ELLIPSOID",
        5 : "ray_intersect_CYLINDER",
        6 : "ray_intersect_BOX"
    }

    def ray_intersect_PLANE(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with a plane
        Params:
            ray_start: 3D vector, the start point of the ray
            ray_dir: 3D vector, the direction of the ray
            geom_position: 3D vector, the position of the plane
            geom_rotation: 3x3 matrix, the rotation of the plane
            geom_size: 3D vector, the size of the plane, half extents in x, y, z: [half_x, half_y, half_z]
        Return:
            hit: boolean, whether the ray intersects with the plane
            hit_pos: 3D vector, the intersection point in the world frame
            hit_distance: float, the distance from the ray_start to the hit
        """
        return hit, hit_pos, hit_distance

    def ray_intersect_SPHERE(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with a sphere
        Params:
            ...
            geom_size: 3D vector, the size of the sphere, radius: [radius, radius, radius]
        Return:
            ...
        """
        return hit, hit_pos, hit_distance

    def ray_intersect_CAPSULE(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with a capsule
        Params: ...
            geom_size: 3D vector, the size of the capsule, radius and half length: [radius, radius, half_length]
        Return: ...
        """
        return hit, hit_pos, hit_distance

    def ray_intersect_ELLIPSOID(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with an ellipsoid
        Params: ...
            geom_size: 3D vector, the size of the ellipsoid, radii: [radius_x, radius_y, radius_z]
        Return: ...
        """
        return hit, hit_pos, hit_distance

    def ray_intersect_CYLINDER(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with a cylinder
        Params: ...
            geom_size: 3D vector, the size of the cylinder, radius and half length: [radius, radius, half_length]
        Return: ...
        """
        return hit, hit_pos, hit_distance

    def ray_intersect_BOX(self, ray_start, ray_dir, geom_position, geom_rotation, geom_size):
        """
        Check if a ray intersects with a box
        Params: ...
            geom_size: 3D vector, the size of the box, half extents in x, y, z: [half_x, half_y, half_z]
        Return: ...
        """
        return hit, hit_pos, hit_distance

    def ray_intersect(self, ray_start, ray_dir, geom):
        """
        Check if a ray intersects with a geom
        Params:
            ray_start: 3D vector, the start point of the ray
            ray_dir: 3D vector, the direction of the ray
            geom: mujoco.MjvGeom object, the geom to intersect with
        Return:
            hit: boolean, whether the ray intersects with the geom
            hit_pos: 3D vector, the intersection point in the world frame
            hit_distance: float, the distance from the ray_start to the hit
        """
        assert ray_start.shape == (3,), "ray_start should be a 3D vector"
        assert ray_dir.shape == (3,), "ray_dir should be a 3D vector"
        assert isinstance(geom, mujoco._structs.MjvGeom), "geom should be a mujoco.MjvGeom object"
        
        geom_type = mujoco.mjtGeom(geom.type)[7:]
        geom_position = geom.pos
        geom_rotation = geom.mat
        geom_size = geom.size

        assert geom_position.shape == (3,), "geom_position should be a 3D vector"
        assert geom_rotation.shape == (3,3), "geom_rotation should be a 3x3 matrix"
        assert geom_size.shape == (3,), "geom_size should be a 3D vector"

        return eval(f"self.{self.mjtGeom2Func[geom_type]}")(ray_start, ray_dir, geom_position, geom_rotation, geom_size)

    def ray_cast(self, rays, sensor_pose, mj_scene):
        """
        Cast rays from sensor_pose to mj_scene and return the intersection points
        Params:
            rays: Nx2 matrix, each row is a ray in the form [phi, thera]
            sensor_pose: 4x4 matrix, the pose of the sensor in the world frame
            mj_scene: mujoco.MjvScene object, the scene to cast rays into
        Return: 
            Nx3 matrix, each row is the intersection point of the corresponding ray locally in the sensor frame

        """
        assert sensor_pose.shape == (4, 4), "sensor_pose should be a 4x4 matrix"
        assert len(rays.shape) == 2 and rays.shape[1] == 2, "rays should be a Nx2 matrix"
        assert isinstance(mj_scene, mujoco._structs.MjvScene), "mj_scene should be a mujoco.MjvScene object"

        hit_points_world = np.zeros((rays.shape[0], 3), np.float32)
        points_intensity = np.zeros(rays.shape[0], np.float32)

        ray_phis = rays[:,0]
        ray_thetas = rays[:,1]
        
        ray_dir = np.zeros((rays.shape[0], 3))
        ray_dir[:,0] = np.cos(ray_thetas) * np.cos(ray_phis)
        ray_dir[:,1] = np.cos(ray_thetas) * np.sin(ray_phis)
        ray_dir[:,2] = np.sin(ray_thetas)
        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=1).reshape(-1,1)

        ray_start = sensor_pose[:3,3]

        for i in range(rays.shape[0]):
            min_hit_dist = 1e10
            for gi in mj_scene.ngeom:
                hit, hit_pos, hit_distance = self.ray_intersect(ray_start, ray_dir, mj_scene.geoms[gi])
                if hit and hit_distance < min_hit_dist:
                    hit_points_world[i] = hit_pos
                    points_intensity[i] = 1.0
        
        tmat_sensor_to_world = np.linalg.inv(sensor_pose)
        out_points_local = tmat_sensor_to_world[:3,:3] @ hit_points_world.T + tmat_sensor_to_world[:3,3].reshape(3,1)
        return out_points_local
    
if __name__ == "__main__":
    pass