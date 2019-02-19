import numpy as np

import datajoint as dj
from tqdm import tqdm
import pandas as pd
import time

import matplotlib.pyplot as plt
import ipyvolume.pylab as p3

ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')

fetched_mesh = (ta3p100.Mesh & ta3p100.CurrentSegmentation & 'segment_id=648518346341366885').fetch1()
# fetched_mesh = (ta3p100.Decimation35 & ta3p100.CurrentSegmentation & 'segment_id=648518346341366885').fetch1()

class Voxels:
    def __init__(self, origin, side_length, offsets, per_voxel_vertices=None):
        self.origin = origin
        self.side_length = side_length
        self.offsets = offsets
        self.per_voxel_vertices = per_voxel_vertices
    
    @property
    def bboxes(self):
        voxel_min = self.origin + (self.offsets * self.side_length)
        voxel_max = voxel_min + self.side_length
        return np.stack([voxel_min, voxel_max], axis=2)
    
    @property
    def centroids(self):
        return self.bboxes.mean(axis=0)
    
    @property
    def volume(self):
        """
        Returns the estimation of volume based on the voxelization. Units are the same as the vertex input.
        """
        return (self.side_length**3) * len(self)
    
    @property
    def _rectangular_idx(self):
        X = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        Y = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        Z = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
        return np.vstack((X, Y, Z))
    
    @property
    def drawable_bboxes(self):
        return self.bboxes[:, np.arange(3), self._rectangular_idx.T].transpose(0, 2, 1)
    
    def plot(self, width=800, height=600, voxel_count_offset=0, voxel_limit=None, use_centroids_instead=False, ipyvol_fig=None, **kwargs):
        """
        :param **kwargs: Is used in ipyvolume.pylab.scatter() or ipyvolume.pylab.plot() depending on whether use_centroids_instead
        is set to true or not.
        """
        if ipyvol_fig is None:
            p3.figure(width=width, height=height)
        else:
            p3.figure(ipyvol_fig)
        voxels_length = len(self)
        if voxel_count_offset >= voxels_length:
            raise ValueError("voxel_count_offset is greater than the number of voxels!")
        else:
            if voxel_limit is None:
                n_voxels_to_plot = len(self)
            else:
                n_voxels_to_plot = voxel_count_offset + voxel_limit
                if n_voxels_to_plot > voxels_length:
                    n_voxels_to_plot = len(self)

        if use_centroids_instead: # I should be careful about forcing the voxels to be divided by 1000 if this is a generalized package
            p3.scatter(self.centroids[voxel_count_offset:n_voxels_to_plot].T/1000, **kwargs)
        else:
            drawable_bboxes = self.drawable_bboxes[voxel_count_offset:n_voxels_to_plot]
            for drawable_bbox in drawable_bboxes:
                p3.plot(*drawable_bbox/1000, **kwargs)
        p3.squarelim()
        p3.show()
    
    def search_by_offset(self, key):
        return np.where((self.offset_vectors==key).all(axis=1))[0]
    
    def to_dict(self, include_origin=True, include_side_length=True, include_offsets=True, include_per_voxel_vertices=False):
        record = dict()
        if include_origin:
            record.update(dict(origin=self.origin))
        if include_side_length:
            record.update(dict(side_length=self.side_length))
        if include_offsets:
            record.update(dict(offsets=self.offsets))
        if include_per_voxel_vertices:
            record.update(dict(per_voxel_vertices=self.per_voxel_vertices))
        return record
    
    def __len__(self):
        return len(self.offsets)

class VoxelMesh:
    def __init__(self, vertices, triangles=None, voxels=None, segment_id=None):
        self.vertices = vertices
        self.triangles = triangles
        if voxels is not None:
            self.voxels = voxels
        self.segment_id = segment_id
        # Won't allow this for now, too many problems it can cause.
        #self.are_vertices_sorted = are_vertices_sorted
    
    @property
    def bbox(self):
        return np.array([(np.min(axis), np.max(axis)) for axis in self.vertices.T])
    
    @property
    def sorted_vertices(self):
        """
        Not actually used for any part of the voxelization, too slow. Here for potential usage
        """
        a = self.vertices
#         if not self.are_vertices_sorted:
        # This still might be slower.
        a = a[a[:,2].argsort()] # First sort doesn't need to be stable.
        a = a[a[:,1].argsort(kind='mergesort')]
        a = a[a[:,0].argsort(kind='mergesort')]
        return a
    
    @property
    def voxels(self):
        return self._voxels
    
    @voxels.setter
    def voxels(self, voxels):
        if isinstance(voxels, Voxels):
            self._voxels = voxels
        else:
            raise TypeError("The voxels property only accepts a Voxels object.")

    def plot(self, plot_mesh=True, plot_voxels=True, width=800, height=600, voxel_count_offset=0,
             voxel_limit=None, use_centroids_instead=False, mesh_color='red', **kwargs):
        """
        :param **kwargs: Is used in ipyvolume.pylab.scatter() or ipyvolume.pylab.plot() depending on whether use_centroids_instead
        is set to true or not.
        """
        if plot_mesh:
            if mesh.triangles is None:
                raise ValueError("There is no face/triangle data stored for this mesh!")
            else:
                fig = p3.figure(width=width, height=height)
                p3.plot_trisurf(*self.vertices.T/1000, self.triangles, color=mesh_color)
                if plot_voxels:
                    self.voxels.plot(width=width, height=height,
                                     voxel_count_offset=voxel_count_offset,
                                     voxel_limit=voxel_limit,
                                     use_centroids_instead=use_centroids_instead,
                                     ipyvol_fig=fig,
                                     **kwargs)
                else:
                    p3.squarelim()
                    p3.show()
        elif plot_voxels:
            self.voxels.plot(width=width, height=height,
                             voxel_count_offset=voxel_count_offset,
                             voxel_limit=voxel_limit,
                             use_centroids_instead=use_centroids_instead,
                             **kwargs)
    
    def voxelize(self, side_length):
        def apply_split(vertices, edges, sort_axis):
            """
            :param vertices: The vertices to sort through and split.
            :param edges: The edges along which to split the array.
            :param sort_axis: The axis to sort and split the array with.
            """
            sorted_verts = vertices[vertices[:,sort_axis].argsort()]
            splitter = sorted_verts[:,sort_axis].searchsorted(edges)
            split = np.array(np.split(sorted_verts, splitter)[1:])
            offset_idx = [i for i, block in enumerate(split) if len(block) > 0]
            # This commented out portion is actually slower, and doesn't seem to work if the voxel size is too small.
            # Should come back to this idea though later.
            #offset_idx = np.unique(edges[1:].searchsorted(sorted_verts).T[sort_axis])
            return np.array((offset_idx, split[offset_idx]))
        
        bbox = self.bbox
        # Get the number of voxels to be used to create cubes (allowing for pushing past the boundaries).
        num_voxels = np.ceil((np.abs(np.subtract(*bbox.T) / side_length))).astype(int)
        
        # Create the cube voxel grid split structure.
        start_coord = bbox.T[0]
        cube_friendly_bbox = np.vstack((start_coord, start_coord + (num_voxels * side_length + 1))).T
        x_edges, y_edges, z_edges = [np.arange(minimum, maximum, side_length) for minimum, maximum in cube_friendly_bbox]
        
        offset_vectors = list()
        voxel_vertices = dict()
                
        # Need to do the initial sort/split on the x_axis
        x_split = np.array(apply_split(self.vertices, x_edges, 0)).T
        for x_id, x_block in x_split:
            
            # Then sort/split through the y_axis
            y_split = np.array(apply_split(x_block, y_edges, 1)).T
            for y_id, y_block in y_split:
                
                # Finally sort/split the z_axis
                z_split = np.array(apply_split(y_block, z_edges, 2)).T
                for z_id, z_block in z_split:
                    
                    # The offset vectors that contain vertices at all are present here
                    key = (x_id, y_id, z_id)
                    offset_vectors.append(key)
                    voxel_vertices[key] = z_block

        offset_vectors = np.array(offset_vectors)
        self.voxels = Voxels(self.bbox[:,0], side_length, offset_vectors, voxel_vertices)

mesh = VoxelMesh(fetched_mesh['vertices'], fetched_mesh['triangles'])

mesh.voxelize(5000)
print(len(mesh.voxels))

mesh.plot(color='blue')