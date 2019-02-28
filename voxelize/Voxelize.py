import numpy as np
from scipy.spatial.distance import cdist

# import pandas as pd # Might want this for pd.unique() at some point? Or for other uses, like the DataFrame for the voxels and to_dict replacement?

# Might want the plotting capabilities to be a separate import-able or enable-able module
# in case matplotlib or ipyvolume are not installed or in a jupyter notebook
# import matplotlib.pyplot as plt
# Also should ensure that the plotting is not specific to a certain set of units (the divide by 1000 to go from nm to um)
import ipyvolume.pylab as p3

# Should also allow the vertices to be accessed per voxel by an np.where set intersection thingy.
# It's more expensive than returning it through the voxelization method that already does it, but it's still useful to have since I likely won't be storing that data in the database.

# Needs to also be able to add and manipulate the existing offsets, though in a less direct way (through a convenient and secure API).
# Should also implement single/many or an amount determinable by the client for voxel accessing of centroids and bboxes.

# I need to implement something where I can query for a voxel, then get its real-coordinate bounding box
# by passing the voxel index or the offset vectors themselves into a function. (Likely have both options).

# Also might want to combine the VoxelMesh class and the Voxels class and just allow it to have options to have all or none of the vertices, triangles,
# and voxels. Then if vertices are present I can produce per_voxel_vertices potentially... That might require another method to also catch the per_voxel_vertices
# though.

# However, combining the two classes would mean I would need a very clear denotation for when a property or function is related to the voxels
# and not the mesh.

# Also need to be able to import already made voxels still.
class Voxels:
    def __init__(self, origin, side_length, offsets, per_voxel_vertices=None):
        """
        Voxel structure for storing and accessing the voxels and optionally the mesh vertices contained within them.
        :param origin: Starting point for which the voxel vectors will be offset from.
        :param side_length: Length of a single side of the stored cube voxels.
        :param offset: An array of (x, y, z) offsets from the origin that are calculated using the side length.
        :param per_voxel_vertices: The vertices contained within each voxel accessible as a dictionary with the voxel offset tuple as its key.
        """
        self.origin = origin
        self.side_length = side_length
        self.offsets = offsets
        self.per_voxel_vertices = per_voxel_vertices

    # Can probably make this into a utility function instead. It is useful in other cases too.
    @property
    def pairwise_differences(self):
        """
        Returns the difference between each offset and the rest of the offsets. There will be a difference array for every voxel offset.
        """
        return np.array([offset - self.offsets for offset in self.offsets])

    @property
    def fast_edges(self):
        def discover_edges(offsets, column_index):
            edge_list = list()
            column_offsets = offsets[:, column_index]
            unique_column = np.unique(column_offsets)
            for column_value in unique_column:
                idx = np.nonzero(column_offsets==column_value)[0]
                portion = offsets[idx]
                pairwise_distances = cdist(portion, portion)
                edges = idx[np.argwhere(pairwise_distances==1)]
                edge_list.append(edges)
            return np.vstack(edge_list)

        edges = list()
        offsets = self.offsets
        for column_index in range(3):
            edges.append(discover_edges(offsets, column_index))
        all_edges = np.vstack(edges)
        all_edges.sort()
        return np.unique(all_edges, axis=0)

    @property
    def adjacency_masks(self):
        """
        Generates masks that are used in most of the adjacency/neighborhood generators.
        """
        return np.abs(self.pairwise_differences).sum(axis=2) == 1

    # Corner adjacency might actually be very helpful in creating a skeleton.
    # It would easily allow me to just connect every voxel by an edge to in the 3x3x3 space surrounding each one.
    # I could likely find corner adjacency by using some metrics to see if the pairwise differences have an absolute value of (1, 1, 1)
    # and those would be the corner adjacencies.
    # If I do it that way I should also move pairwise_differences to its own property.
    @property
    def adjacency(self): # If I include corner adjacency I should have this property give face-to-face and corner-to-corner adjacency
                         # and then have 2 more properties to return those separately.
        """
        Returns the face-face adjacency matrix of the voxels (based on the offset array). As of right now it only fills the upper triangle
        """
        return self.adjacency_masks.astype(np.uint8)

    # Or I can just avoid having this as a property at all. That seems like a better idea, because otherwise if multiple steps are rerun at different
    # times then the data integrity might not hold up.

    # Because this stuff has to rerun everything, maybe I should turn all of these into functions that also store these things as private variables?
    # Would speed up re-use and keep the data integrity.
    @property
    def edges(self):
        """
        Convenience method to computes the edges of the adjacency matrix (recomputes the adjacency matrix). This is a wrapper around get_edges() so it is preferable to use the method instead due to the property recomputing the adjacency matrix each time.
        """
        return self.get_edges(self.adjacency)

    @property
    def neighborhood(self):
        """
        Generates the voxel neighborhood as a dictionary with the offset as a key and the neighboring offsets as its values.
        """
        return {tuple(self.offsets[i]): self.offsets[mask] for i, mask in enumerate(self.adjacency_masks)}
    
    # Might want a version that returns it as an ndarray?
    @property
    def idx_neighborhood(self):
        """
        Generates the voxel neighborhood as a dictionary with the voxel index as a key and the neighboring voxel indexes as its values.
        """
        return {i: np.where(mask)[0] for i, mask in enumerate(self.adjacency_masks)}

    @property
    def bboxes(self):
        """
        Conversion of the voxel offset structure to their real-world cube bounding boxes.
        """
        voxel_min = self.origin + (self.offsets * self.side_length)
        voxel_max = voxel_min + self.side_length
        return np.stack([voxel_min, voxel_max], axis=2)
    
    @property
    def centroids(self):
        """
        Centroid coordinates of the bounding boxes of the cube voxels.
        """
        return self.bboxes.mean(axis=2)
    
    @property
    def com(self):
        """
        Centroid or center of mass (com) for entire voxel structure.
        """
        return self.centroids.mean(axis=0)

    @property
    def vertex_centroids(self):
        """
        Center of mass of the vertices per voxel. This version requires per_voxel_vertices to be defined and filled. Maybe this is better called the center of gravity? But it really is a center of actual mass.
        """
        return np.array([vertex_block.mean(axis=0) for vertex_block in self.per_voxel_vertices.values()])

    # Important Thresholding Idea for the Voxelization:
    # - I could threshold and remove the outer voxels that have the bbox of their vertices only present
    #  in less than a certain percentage of the voxel volume! This would allow axons to not be thresholded
    #  out if they run through a majority of the voxel (though there does come a problem if an axon runs through
    #  the border between two voxels. Doing this type of thresholding might require the additional support of corner
    #  adjacency (if certain connecting voxels get removed, which could still be a problem in certain cases even without thresholding).
    @property
    def volume(self):
        """
        Returns the estimation of volume based on the voxelization. Units are the same as the vertex input.
        """
        return (self.side_length**3) * len(self)
    
    # Voxel mesh skin:
    # - Perhaps I should just put a shrink wrap around the Mesh.
    # - Or I could just connect all the outer voxels (but actually translate them to the centroids,
    #  or even better the vertex centers at some point that doesn't mess up the convenience of the voxels)
    #  and then for every "square" I just very simply split it in two. The triangles on faces that have an edge
    #  would then be deleted (or at that point I can simply not create triangles with them). Otherwise, the voxels
    #  that jut out and wouldn't form squares would very simply form triangles naturally. Might have to figure out how
    #  to query for corner adjacency instead of face-to-face though. Also I have to deal with single voxel lines that exist
    #  because of say axons. Maybe just split every voxel's faces into a series of triangles (while not bothering with connecting
    #  corner adjacent voxels) and remove all "non-visible" or outer triangles.
    # In that case it might be helpful to use the bounding box of the vertices (if I'm unable to use the vertex centers of mass per voxel).
    # Once the mesh is watertight and manifold, I should be able to do another quick passthrough to remove internal faces (like if there are pockets of air
    # inside the mesh caused by 2 layers of faces).
    
    def vbbox(self, offset):
        voxel_min = self.origin + (offset * self.side_length)
        voxel_max = voxel_min + self.side_length
        return np.array((voxel_min, voxel_max)).T

    def voxel_bbox(self, voxel_id):
        offset = self.offsets[voxel_id]
        return self.vbbox(offset)
    

    @staticmethod
    def get_edges(adjacency_array):
        """
        Returns the edge connections of an argument adjacency matrix.
        """
        return np.argwhere(adjacency_array)

    def get_edge_directions(self, edges):
        """
        Returns the direction the edges are oriented in.
        """
        return np.subtract(*self.offsets[edges].transpose(1, 0, 2))

    def get_edge_vectors(self, edges):
        """
        Returns the real-world vectors of the edges. Though should it be center to center? Yeah probably.
        """
        raise NotImplementedError
    
    @property
    def _voxel_vertex_idx(self):
        return np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])

    @property
    def offset_vertices(self):
        """
        Generates the vertices at the corner of each voxel in units of offset.
        """
        return np.array((self.offsets, self.offsets+1)).transpose(1, 2, 0)[:, np.arange(3), self._voxel_vertex_idx]

    @property
    def _offset_vertices_to_faces_idx(self):
        """
        Used for converting from the corner vertices of each voxel to faces.
        """
        return np.array([[0, 3, 2, 1],
                         [0, 1, 5, 4],
                         [0, 4, 7, 3],
                         [1, 2, 6, 5],
                         [3, 7, 6, 2],
                         [4, 5, 6, 7]])

    @property
    def triangular_voxel_mesh(self):
        """
        Generates a triangular mesh from the visible faces of the voxels.
        """
        adj_mat = self.adjacency
        edges = self.get_edges(np.tril(adj_mat))
        edge_directions = self.get_edge_directions(edges)
        offset_vertices = self.offset_vertices
        unique_voxel_vertices, flat_inverse = np.unique(offset_vertices.reshape(-1, 3), axis=0, return_inverse=True)
        new_voxel_vertices = self.origin + (unique_voxel_vertices * self.side_length)
        vertices_inverse = flat_inverse.reshape(-1, 8)

        offset_vertices_to_faces_idx = self._offset_vertices_to_faces_idx

        remaining_faces = dict()
        face_idx_master = np.arange(6)
        for i, (direction, vertices) in enumerate(zip(edge_directions, offset_vertices)):
            edge_idx = np.where(i == edges)
            face_idx = face_idx_master.copy() # Faster to copy than to reinitialize using np.arange each time.
            remove_face_idx = list()
            for idx, direction in zip(edge_idx[1], edge_directions[edge_idx[0]]):
                dir_idx = np.where(direction==1)[0].item()
                if idx == 0:
                    if dir_idx == 0:
                        remove_face_idx.append(2)
                    elif dir_idx == 1:
                        remove_face_idx.append(1)
                    elif dir_idx == 2:
                        remove_face_idx.append(0)
                elif idx == 1:
                    if dir_idx == 0:
                        remove_face_idx.append(3)
                    elif dir_idx == 1:
                        remove_face_idx.append(4)
                    elif dir_idx == 2:
                        remove_face_idx.append(5)
            remaining_faces[i] = vertices_inverse[i][offset_vertices_to_faces_idx][np.delete(face_idx, remove_face_idx)]
            
        new_voxel_triangles = list()
        for voxel_id, faces in remaining_faces.items():
            if len(faces) > 0:
                for face in faces:
                    triangle1 = face[[0, 1, 2]]
                    triangle2 = face[[2, 3, 0]]
                    new_voxel_triangles.extend((triangle1, triangle2))
        new_voxel_triangles = np.array(new_voxel_triangles)
        return new_voxel_vertices, new_voxel_triangles

    @property
    def _cube_face_idx(self):
        """
        Used to convert a voxel bounding box to its 6 faces.
        """
        # Implemented this way to save on space.
        return np.array([[0, 0, 1, 1,  0, 1, 1, 0,  0, 0, 0, 0,  1, 1, 1, 1,  0, 0, 1, 1,  0, 1, 1, 0],
                         [0, 1, 1, 0,  0, 0, 0, 0,  0, 0, 1, 1,  0, 1, 1, 0,  1, 1, 1, 1,  0, 0, 1, 1],
                         [0, 0, 0, 0,  0, 0, 1, 1,  0, 1, 1, 0,  0, 0, 1, 1,  0, 1, 1, 0,  1, 1, 1, 1]]).T.reshape(6, 4, 3)

    # This could actually be a function that takes in whatever voxels wanted.
    @property
    def voxel_faces(self):
        """
        Faces of the voxel bounding boxes.
        """
        return self.bboxes[:, np.arange(3), self._cube_face_idx]

    @property
    def _drawable_idx(self):
        """
        Used to convert a voxel bounding box to a line plottable feature in ipyvolume.
        """
        X = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        Y = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        Z = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
        return np.vstack((X, Y, Z))
    
    @property
    def drawable_bboxes(self):
        """
        Voxel bounding boxes as line plottable arrays.
        """
        return self.bboxes[:, np.arange(3), self._drawable_idx.T].transpose(0, 2, 1)

    # I need to be able to extract the per voxel vertices from a mesh when I already have offsets imported from the database.


    # Also have it except then raise an error if the offsets don't exist.

    # Need to be able to also plot the center of mass of the vertices within each voxel as an option instead of the centroids.
    def plot(self, width=800, height=600, voxel_count_offset=0, voxel_limit=None, use_centroids_instead=False, ipyvol_fig=None, scaling=1, **kwargs):
        """
        This method needs better documentation.

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

        if use_centroids_instead:
            if 'marker' not in kwargs:
                kwargs['marker'] = 'sphere'
            if 'color' not in kwargs:
                kwargs['color'] = 'blue'
            if 'size' not in kwargs:
                kwargs['size'] = 0.5
            p3.scatter(*self.centroids[voxel_count_offset:n_voxels_to_plot].T*scaling, **kwargs)
        else:
            drawable_bboxes = self.drawable_bboxes[voxel_count_offset:n_voxels_to_plot]
            for drawable_bbox in drawable_bboxes:
                p3.plot(*drawable_bbox*scaling, **kwargs)
        p3.squarelim()
        p3.show()
    
    # Maybe want to allow it to take in a list and return indices for all of the voxels in that argument.
    def search_by_offset(self, key):
        """
        Searches within the offset array to find the index at which a specific voxel resides.
        :param key: An (a, b, c) tuple-like structure where each value corresponds to the x, y, and z offsets respectively.
        :returns: Index of a specific voxel.
        """
        return np.where((self.offsets==key).all(axis=1))[0]
    
    def to_dict(self, include_origin=True, include_side_length=True, include_offsets=True, include_per_voxel_vertices=False):
        """
        Returns the important attributes of the Voxels object as a dictionary.
        """
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
    
    # Maybe shouldn't override since it's just for offsets
    def __len__(self):
        return len(self.offsets)
    
# I should additionally add a flag to exclude isolated vertices from voxelization, which will also require the triangles.
class VoxelMesh:
    def __init__(self, vertices, triangles=None, voxels=None, segment_id=None, exclude_isolated_vertices=False):
        """
        Mesh structure that handles voxelization.
        :param vertices: Vertex coordinates in x, y, z space of the mesh.
        :param triangles: Triangle that each reference 3 vertices by index.
        :param voxels: Voxels object, can either be added at initialization or they can be generated by this class using the voxelize method.
        :param segment_id: Optional additional identification for the VoxelMesh object. Can be used at your disrection but will eventually also allow extra query functionality when using the VoxelStatistics module.
        :param exclude_isolated_vertices: Optional flag which requires the triangle data to exclude vertices not referenced by the triangles.
        """
        self.vertices = vertices
        self.triangles = triangles
        if voxels is not None:
            self.voxels = voxels
        self.segment_id = segment_id
        self.exclude_isolated_vertices = exclude_isolated_vertices
        # Won't allow this for now, too many problems it can cause.
        #self.are_vertices_sorted = are_vertices_sorted
    
    @property
    def bbox(self):
        """
        Bounding box of the mesh (based on its vertices). Isolated verties are NOT excluded.
        """
        return self.get_bbox(self.vertices)

    @property
    def exclude_isolated_vertices(self):
        """
        Optional flag which requires the triangle data to exclude vertices not referenced by the triangles.
        """
        return self._exclude_isolated_vertices

    @exclude_isolated_vertices.setter
    def exclude_isolated_vertices(self, flag):
        if flag:
            if self.triangles is None:
                # Should make the exception more specific.
                raise TypeError("exclude_isolated_vertices flag can only be changed is triangle data is available for the mesh.")
            else:
                # Clearer execution logic to explicitly assign as True rather than use flag and ensures a proper bool is saved.
                self._exclude_isolated_vertices = True
        else:
            self._exclude_isolated_vertices = False

    @property
    def nonisolated_vertices(self):
        """
        Uses the triangle data to remove isolated vertices. Does not preserve or fix triangle indices. That might be implemented later, it can take a few seconds to compute.
        """
        return self.vertices[self.triangles]

    @property
    def center_of_triangles(self):
        """
        Returns the centroids of the triangles, can potentially be used as a replacement or addition to the vertices to alter the voxelization depending on how large the triangles are and how spaced out the vertices are.
        """
        return self.vertices[self.triangles].mean(axis=1)

    # Would np.array(np.where(mesh.vertices)).T potentially give me the sorted vertices, and faster?
    @property
    def sorted_vertices(self):
        """
        Not actually used for any part of the voxelization, too slow. Here for potential usage, but it sorts the vertices by first column, then second, then third while preserving the row value assignments.
        """
        a = self.vertices
#         if not self.are_vertices_sorted:
        # This still might be slower even just as the initial sort than the entire voxelization process is. Also not implemented to properly use this for voxelization.
        a = a[a[:,2].argsort()] # First sort doesn't need to be stable.
        a = a[a[:,1].argsort(kind='mergesort')]
        a = a[a[:,0].argsort(kind='mergesort')]
        return a
    
    @property
    def voxels(self):
        """
        Voxels object, which contains an origin, side length, offset vectors, and optionally the real-world vertices accesible per voxel as a dictionary.
        """
        return self._voxels
    
    @voxels.setter
    def voxels(self, voxels):
        # """
        # Voxels object, which contains an origin, side length, offset vectors, and optionally the real-world vertices accesible per voxel as a dictionary.
        # """
        if isinstance(voxels, Voxels):
            self._voxels = voxels
        else:
            raise TypeError("The voxels property only accepts a Voxels object.")

    @staticmethod
    def get_bbox(vertices):
        return np.array([(np.min(axis), np.max(axis)) for axis in vertices.T])

    def plot(self, plot_mesh=True, plot_voxels=True, width=800, height=600, voxel_count_offset=0,
             voxel_limit=None, use_centroids_instead=False, scaling=1, mesh_color='red', **kwargs):
        """
        This method needs better documentation.

        :param **kwargs: Is used in ipyvolume.pylab.scatter() or ipyvolume.pylab.plot() depending on whether use_centroids_instead
        is set to true or not.
        """
        if plot_mesh:
            if self.triangles is None:
                raise ValueError("There is no triangle data stored for this mesh!")
            else:
                fig = p3.figure(width=width, height=height)
                p3.plot_trisurf(*self.vertices.T*scaling, self.triangles, color=mesh_color)
                if plot_voxels:
                    self.voxels.plot(width=width, height=height,
                                     voxel_count_offset=voxel_count_offset,
                                     voxel_limit=voxel_limit,
                                     use_centroids_instead=use_centroids_instead,
                                     ipyvol_fig=fig,
                                     scaling=scaling,
                                     **kwargs)
                else:
                    p3.squarelim()
                    p3.show()
        elif plot_voxels:
            try:
                fig = p3.figure(width=width, height=height)
                self.voxels.plot(width=width, height=height,
                                 voxel_count_offset=voxel_count_offset,
                                 voxel_limit=voxel_limit,
                                 use_centroids_instead=use_centroids_instead,
                                 ipyvol_fig=fig,
                                scaling=scaling,
                                **kwargs)
            except AttributeError:
                raise AttributeError("This object does not have a Voxels object, you must initialize the voxel mesh with a Voxels object or run voxelize() to generate new voxels.")
    
    # To deal with the cases of very small voxel sizes, I likely need to implement something to throw out large chunks of voxels as a quick preprocessing step before the main voxelization.
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
        
        if self.exclude_isolated_vertices:
            vertices = self.nonisolated_vertices
            bbox = self.get_bbox(vertices)
        else:
            vertices = self.vertices
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
        x_split = np.array(apply_split(vertices, x_edges, 0)).T
        for x_id, x_block in x_split:
            
            # Then sort/split through the y_axis
            y_split = np.array(apply_split(x_block, y_edges, 1)).T
            for y_id, y_block in y_split:
                
                # Finally sort/split the z_axis
                z_split = np.array(apply_split(y_block, z_edges, 2)).T
                for z_id, z_block in z_split:
                    
                    # The offset vectors that contain vertices at all are present here and are inserted into the structures
                    key = (x_id, y_id, z_id)
                    offset_vectors.append(key)
                    voxel_vertices[key] = z_block

        # Initialize and store the Voxels object.
        offset_vectors = np.array(offset_vectors)
        self.voxels = Voxels(self.bbox[:,0], side_length, offset_vectors, voxel_vertices)
