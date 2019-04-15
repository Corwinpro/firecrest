import os
import pygmsh as pg
import meshio
import numpy as np
import subprocess
import warnings
from abc import ABC, abstractmethod
import dolfin as dolf
from dolfin_utils.meshconvert import meshconvert

# from dolfin_utils.meshconvert import meshconvert
# import dolfin as dolf

GEO_EXT = ".geo"
MSH_EXT = ".msh"
MSH_FORMAT_XDMF = "xdmf"
MSH_FORMAT_XML = "xml"
LOG_EXT = ".log"
GMSH_PATH = ""


def newfolder(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class Geometry(ABC):
    """
    Abstract base class for Geometry Entity.
    A Geometry consists of a set of boundary_elements.
    params:
        - mesh_name: name of mesh files, associated with the geometry
        - mesh_folder: default mesh files directory
        - dimensions: dimensionality of the domain
        - msh_format: XML or XDMF mesh format
    
    attributes:
        - boundary_elements: collection of boundary_elements forming the Geometry
        - pg_geometry: pygmsh generated geometry
        - pg_*: pygmsh generated entities
    """

    def __init__(
        self,
        boundary_elements,
        mesh_name="mesh",
        mesh_folder="Mesh",
        dimensions=2,
        msh_format=MSH_FORMAT_XML,
    ):
        self.boundary_elements = boundary_elements
        self.mesh_folder = mesh_folder
        if self.mesh_folder:
            newfolder(self.mesh_folder)
        self.mesh_name = self.mesh_folder + "/" + mesh_name
        self.dimensions = dimensions
        self.geo_file = self.mesh_name + GEO_EXT
        self.msh_file = self.mesh_name + MSH_EXT
        self.log_file = self.mesh_name + LOG_EXT

        if msh_format not in (MSH_FORMAT_XDMF, MSH_FORMAT_XML):
            warnings.warn(
                "Mesh format {} not supported. Using {} instead.".format(
                    msh_format, MSH_FORMAT_XML
                )
            )
            msh_format = MSH_FORMAT_XML
        self.msh_format = msh_format

        self.dolf_file = self.mesh_name + "." + self.msh_format
        self.pg_geometry = pg.built_in.Geometry()
        self.mark_boundaries()

        self.mesh = None
        self._boundary_parts = None
        self._ds = None
        self._dx = None

    def geo_to_mesh(self):
        """
        command line convertion of a 
            self.geo_file with .geo extension
        to 
            self.msh_file with .msh extension
        """
        cmd = [
            "gmsh",
            self.geo_file,
            "-{}".format(self.dimensions),
            "-format",
            "msh2",
            # "-save_all", # : Save all elements (discard physical group definitions)
            "-o",
            self.msh_file,
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmsh_out, gmsh_err = p.communicate()
        gmsh_out = gmsh_out.decode()
        gmsh_err = gmsh_err.decode()

        # Save output to file
        with open(self.log_file, "w+") as mesh_log:
            mesh_log.write(gmsh_out)
            mesh_log.write(gmsh_err)

    def compile_mesh(self, mesh_format=None):
        """
        Full mesh compilation routine. Geo -> Msh -> Dolf Readable format.
        First, creates a .msh file from .geo using self.geo_to_mesh() method,
        Second, creates an .xdmf or .xml dolfin-readable file self.dolf_file
        """
        if mesh_format is None:
            mesh_format = self.msh_format
        else:
            mesh_format = MSH_FORMAT_XDMF
        self.write_geom()
        self.geo_to_mesh()

        if self.dimensions == 3 and mesh_format == "xdmf":
            mesh_format = "xml"

        if mesh_format == "xml":
            self._convert_xml_to_dolfin()
        elif mesh_format == "xdmf":
            self._convert_xdmf_to_dolfin()
        else:
            raise ValueError("Unknown mesh format: {}".format(mesh_format))

    def _convert_xml_to_dolfin(self):
        try:
            meshconvert.convert2xml(self.msh_file, self.dolf_file)
        except FileNotFoundError:
            return False

    def _convert_xdmf_to_dolfin(self):
        mesh = meshio.read(self.msh_file)

        if self.dimensions == 2:
            # from gist: https://gist.github.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed
            mesh.points = mesh.points[:, :2]  # remove z-values to force 2d mesh
            mesh = meshio.Mesh(
                points=mesh.points, cells={"triangle": mesh.cells["triangle"]}
            )
        elif self.dimensions == 3:
            # TODO: figure out how to fix this
            mesh = meshio.Mesh(points=mesh.points, cells={"tetra": mesh.cells["tetra"]})

        meshio.write(self.dolf_file, mesh)

    def write_geom(self, mesh_name=None):
        """
        Write the geom code to the specified file name
        :param mesh_name: name of the mesh
        """
        if mesh_name is not None:
            file_name = mesh_name + GEO_EXT
        else:
            file_name = self.geo_file

        # First make sure mesh folder exists.
        newfolder(self.mesh_folder)

        with open(file_name, "w") as gmsh_output:
            gmsh_output.write(self.pg_geometry.get_code())

    def load_mesh(self, mesh_name=None, mesh_format=None):
        """
        Reads mesh_name.(xml|xdmf) file and returns a dolfin.Mesh() object
        """
        if mesh_name is None:
            mesh_name = self.mesh_name

        if mesh_format is None:
            mesh_format = self.msh_format
        else:
            mesh_format = MSH_FORMAT_XDMF

        mesh_file = mesh_name + "." + mesh_format

        if mesh_format == "xml":
            mesh = dolf.Mesh(mesh_file)
        elif mesh_format == "xdmf":
            try:
                MPI_COMM = dolf.MPI.comm_world
            except AttributeError as e:
                raise e

            mesh = dolf.Mesh()
            xdmf_file = dolf.XDMFFile(MPI_COMM, mesh_file)
            xdmf_file.read(mesh)
        else:
            raise ValueError("Unknown mesh format: {}".format(mesh_format))

        return mesh

    def mark_boundaries(self):
        """
        The dictionary 'markers_dict' stores the pairs of 
        (keys) boundary type 'btype',
        (values) list of boundary_elements of this boundary type,
        i.e.
        ```
        self.market_dict = {"noslip": [noslip_boundary_1, noslip_boundary_2],
                            "slip": [slip_boundary_3]}
        ```
        """
        self.markers_dict = {}
        for boundary_element in self.boundary_elements:
            if boundary_element.btype in self.markers_dict:
                self.markers_dict[boundary_element.btype].append(boundary_element)
            else:
                self.markers_dict[boundary_element.btype] = [boundary_element]

    @property
    def boundary_parts(self):
        """
        Creates dolfin MeshFunction of dim-1 dimension
        """
        if self._boundary_parts:
            return self._boundary_parts

        assert self.mesh, "Need mesh to be generated to get the boundary parts"

        boundary_parts = dolf.MeshFunction(
            "size_t", self.mesh, self.mesh_name + "_facet_region.xml"
        )

        self._boundary_parts = boundary_parts
        return boundary_parts

    @property
    def ds(self):
        """
        Creates dolfin 'ds' measure, accounting for physical marking
        """
        assert self.mesh, "Mesh needs to be generated"

        if not self._ds:
            self._ds = dolf.Measure(
                "ds", domain=self.mesh, subdomain_data=self.boundary_parts
            )

        return self._ds

    @property
    def dx(self):
        """
        Creates dolfin 'dx' measure
        """
        assert self.mesh, "Mesh needs to be generated"

        if not self._dx:
            self._dx = dolf.dx(domain=self.mesh)

        return self._dx

    def _get_boundaries(self, boundary_type):
        """
        Returns all boundaries of the given boundary_type: str
        """
        return self.markers_dict.get(boundary_type, None)

    def get_boundary_measure(self, boundary_type=None, boundary=None):
        """
        Returns the dolfin 'ds' boundary measure for all the boundaries of the
        - boundary_type: str
        - boundary: BoundaryElement
        If neither is provided, returns the whole boundary measure
        """
        if boundary_type:
            target_boundaries = self._get_boundaries(boundary_type)
            return self.ds(
                tuple(boundary.surface_index for boundary in target_boundaries)
            )
        elif boundary:
            return self.ds(boundary.surface_index)
        else:
            return self.ds


class SimpleDomain(Geometry):
    def __init__(
        self,
        boundary_elements,
        mesh_name="mesh",
        mesh_folder="Mesh",
        dimensions=2,
        **kwargs
    ):
        super().__init__(
            boundary_elements, mesh_name, mesh_folder, dimensions, **kwargs
        )
        self._generate_pg_geometry()
        self.compile_mesh()
        self.mesh = self.load_mesh()

    def _generate_surface_points(self):
        """
        For all boundary elements, we collect the surface points of each
        boundary element and create geo_points list.
        geo_points list is a list of pygmsh Point objects.
        """
        surface_points = []
        geo_points = []
        for boundary_element in self.boundary_elements:
            for position in boundary_element.surface_points:
                if len(surface_points) > 0 and position == surface_points[0]:
                    lcar = min(geo_points[0].lcar, boundary_element.el_size)
                    geo_points[0] = self.pg_geometry.add_point(
                        [position[0], position[1], 0.0], lcar=lcar
                    )
                elif len(surface_points) > 0 and position == surface_points[-1]:
                    lcar = min(geo_points[-1].lcar, boundary_element.el_size)
                    geo_points[-1] = self.pg_geometry.add_point(
                        [position[0], position[1], 0.0], lcar=lcar
                    )
                else:
                    surface_points.append(position)
                    geo_points.append(
                        self.pg_geometry.add_point(
                            [position[0], position[1], 0.0],
                            lcar=boundary_element.el_size,
                        )
                    )
        return geo_points

    def _generate_pg_points(self, boundary_element):
        """
        For the given boundary element, we collect the surface points and create pg_points list.
        We also write the pg_points code to the self.pg_geometry code.
        
        pg_points: list of pygmsh Point objects.
        """
        pg_points = []
        for position in boundary_element.surface_points:
            pg_points.append(
                self.pg_geometry.add_point(
                    [position[0], position[1], 0.0], lcar=boundary_element.el_size
                )
            )

        return pg_points

    def _generate_pg_lines(self, boundary_element):
        """
        Provided the boundary_element instance has a list of pg_points from _generate_pg_points,
        we generate a list of connections (lines) between the pg_points.
        We also write the pg_lines code to the self.pg_geometry code.
        The lines are pygmsh objects Line.
        """
        pg_lines = []
        for i in range(len(boundary_element.pg_points) - 1):
            line = self.pg_geometry.add_line(
                boundary_element.pg_points[i], boundary_element.pg_points[i + 1]
            )
            pg_lines.append(line)
        self.pg_geometry.add_physical_line(
            pg_lines, label=boundary_element.surface_index
        )

        return pg_lines

    def _generate_pg_geometry(self):
        """
        Main SimpleDomain mesh generation and compilation method.
        We prepare pg_points and pg_lines, create a pg_lineloop,
        and finally add a pg_surface bounded by the lineloop.
        All the non-point objects also are gmsh 'physical' objects.
        """
        self.pg_points = []
        self.pg_lines = []

        # Generating pg_points for all boundary elements
        for boundary_element in self.boundary_elements:
            boundary_element.pg_points = self._generate_pg_points(boundary_element)

            self.pg_points.extend(boundary_element.pg_points)

        # Fixing multiple point objects representing same physical point
        # TODO: dict structure here might be more useful?
        for i in range(len(self.boundary_elements) - 1):
            """
            Iterate over the boundary_elements' pg_points, and connect the 
            boundary elemements with index i and i-1 through the left edge point,
            boundary elemements with index i and i+1 through the right edge point.
            """
            if (
                self.boundary_elements[i].el_size
                > self.boundary_elements[i - 1].el_size
            ):
                _left_point = self.boundary_elements[i - 1].pg_points[-1]
            else:
                _left_point = self.boundary_elements[i].pg_points[0]
            self.boundary_elements[i].pg_points[0] = self.boundary_elements[
                i - 1
            ].pg_points[-1] = _left_point

            if (
                self.boundary_elements[i].el_size
                > self.boundary_elements[i + 1].el_size
            ):
                _right_point = self.boundary_elements[i + 1].pg_points[0]
            else:
                _right_point = self.boundary_elements[i].pg_points[-1]
            self.boundary_elements[i].pg_points[-1] = self.boundary_elements[
                i + 1
            ].pg_points[0] = _right_point

        # Generating pg_lines for all boundary elements
        for boundary_element in self.boundary_elements:
            boundary_element.pg_lines = self._generate_pg_lines(boundary_element)
            self.pg_lines.extend(boundary_element.pg_lines)

        # Generating pg_lineloop from pg_lines
        self.pg_lineloop = self.pg_geometry.add_line_loop(self.pg_lines)

        # Generating pg_surface from pg_linespg_lineloop
        self.pg_surface = self.pg_geometry.add_plane_surface(self.pg_lineloop)
        self.pg_geometry.add_physical_surface(self.pg_surface, 0)

