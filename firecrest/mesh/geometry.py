import os
import pygmsh as pg
import meshio
import numpy as np
import subprocess
from abc import ABC, abstractmethod
import dolfin as dolf

# from dolfin_utils.meshconvert import meshconvert
# import dolfin as dolf

GEO_EXT = ".geo"
MSH_EXT = ".msh"
MSH_FORMAT = "xdmf"
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
    """

    def __init__(
        self, boundary_elements, mesh_name="mesh", mesh_folder="Mesh", dimensions=2
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
        self.dolf_file = self.mesh_name + "." + MSH_FORMAT

        self.pg_geometry = pg.built_in.Geometry()

        self.mark_boundaries()

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

    def compile_mesh(self, mesh_format=MSH_FORMAT):
        """
        Full mesh compilation routine.
        First, creates a .msh file from .geo using self.geo_to_mesh() method,
        Second, creates an .xdmf or .xml dolfin-readable file self.dolf_file
        """
        self.write_geom()
        self.geo_to_mesh()

        if self.dimensions == 3 and mesh_format == "xdmf":
            mesh_format = "xml"

        if mesh_format == "xml":
            try:
                pass  # meshconvert.convert2xml(self.msh_file, self.dolf_file)
            except FileNotFoundError:
                return False
        elif mesh_format == "xdmf":
            mesh = meshio.read(self.msh_file)

            if self.dimensions == 2:
                # from gist: https://gist.github.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed
                mesh.points = mesh.points[:, :2]  # remove z-values to force 2d mesh
                mesh = meshio.Mesh(
                    points=mesh.points, cells={"triangle": mesh.cells["triangle"]}
                )
            elif self.dimensions == 3:
                # TODO: figure out how to fix this
                mesh = meshio.Mesh(
                    points=mesh.points, cells={"tetra": mesh.cells["tetra"]}
                )

            meshio.write(self.dolf_file, mesh)
        else:
            raise ValueError("Unknown mesh format: {}".format(mesh_format))

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

    def load_mesh(self, mesh_name=None, mesh_format=MSH_FORMAT):
        if mesh_name is None:
            mesh_name = self.mesh_name

        mesh_file = mesh_name + "." + mesh_format

        if mesh_format == "xml":
            mesh = dolf.Mesh(mesh_file)
        elif mesh_format == "xdmf":
            try:
                MPI_COMM = dolf.MPI.comm_world
            except AttributeError:
                MPI_COMM = dolf.mpi_comm_world()

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


class SimpleDomain(Geometry):
    def __init__(
        self, boundary_elements, mesh_name="mesh", mesh_folder="Mesh", dimensions=2
    ):
        super().__init__(boundary_elements, mesh_name, mesh_folder, dimensions)
        self.generate_geometry()
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

    def _generate_surface_lines(self):
        """
        Provided the instance has a list of geo_points from _generate_surface_points,
        we generate a list of connections (lines) between the geometric points.
        The lines are pygmsh objects Line.
        """
        geo_lines = []
        for i in range(len(self.geo_points) - 1):
            geo_lines.append(
                self.pg_geometry.add_line(self.geo_points[i], self.geo_points[i + 1])
            )
        # Do I need this last connection?
        geo_lines.append(
            self.pg_geometry.add_line(self.geo_points[-1], self.geo_points[0])
        )

        return geo_lines

    def generate_geometry(self):
        """
        Main SimpleDomain mesh generation and compilation method.
        We prepare pygmsh points and lines, create a lineloop,
        and finally add a surface bounded by the lineloop.
        """
        self.geo_points = self._generate_surface_points()
        self.geo_lines = self._generate_surface_lines()

        lineloop = self.pg_geometry.add_line_loop(self.geo_lines)
        self.pg_geometry.add_plane_surface(lineloop)
