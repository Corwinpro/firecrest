import os
import pygmsh as pg
import meshio
import numpy as np
import subprocess
import itertools
from abc import ABC, abstractmethod

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
    def __init__(self, mesh_name="mesh", mesh_folder="Mesh", dimensions=2):
        self.mesh_folder = mesh_folder
        if self.mesh_folder:
            newfolder(self.mesh_folder)
        self.mesh_name = self.mesh_folder + "/" + mesh_name
        self.dimensions = dimensions
        self.geo_file = self.mesh_name + GEO_EXT
        self.msh_file = self.mesh_name + MSH_EXT
        self.log_file = self.mesh_name + LOG_EXT
        self.dolf_file = self.mesh_name + "." + MSH_FORMAT

        self.geometry = pg.built_in.Geometry()

    def geo_to_mesh(self):
        """
        Converts a 
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
        """Write the geom code to the specified file name
        :param mesh_name: name of the mesh
        """
        if mesh_name is not None:
            file_name = mesh_name + GEO_EXT
        else:
            file_name = self.geo_file

        # First make sure mesh folder exists.
        newfolder(self.mesh_folder)

        with open(file_name, "w") as gmsh_output:
            gmsh_output.write(self.geometry.get_code())

    def load_mesh(self, mesh_name=None, mesh_format=MSH_FORMAT):
        if mesh_name is None:
            mesh_name = self.mesh_name

        mesh_file = mesh_name + "." + mesh_format
        """
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
        """
        mesh = None
        return mesh

    @abstractmethod
    def _generate_surface_points(self):
        pass

    @abstractmethod
    def _generate_surface_lines(self):
        pass

    def generate_geometry(self):
        self.geo_points = self._generate_surface_points()
        self.geo_lines = self._generate_surface_lines()

        lineloop = self.geometry.add_line_loop(self.geo_lines)
        self.geometry.add_plane_surface(lineloop)

        self.compile_mesh()

        # self.mesh = dolf.Mesh(self.mesh_name + MSH_FORMAT)


class SimplyConnected(Geometry):
    def __init__(
        self, boundary_elements, mesh_name="mesh", mesh_folder="Mesh", dimensions=2
    ):
        super().__init__(mesh_name, mesh_folder, dimensions)
        self.boundary_elements = boundary_elements
        self.generate_geometry()

    def _generate_surface_points(self):
        surface_points = []
        geo_points = []
        for boundary_element in self.boundary_elements:
            for position in boundary_element.surface_points:
                if len(surface_points) > 0 and position == surface_points[0]:
                    lcar = min(geo_points[0].lcar, boundary_element.el_size)
                    geo_points[0] = self.geometry.add_point(
                        [position[0], position[1], 0.0], lcar=lcar
                    )
                elif len(surface_points) > 0 and position == surface_points[-1]:
                    lcar = min(geo_points[-1].lcar, boundary_element.el_size)
                    geo_points[-1] = self.geometry.add_point(
                        [position[0], position[1], 0.0], lcar=lcar
                    )
                else:
                    surface_points.append(position)
                    geo_points.append(
                        self.geometry.add_point(
                            [position[0], position[1], 0.0],
                            lcar=boundary_element.el_size,
                        )
                    )
        return geo_points

    def _generate_surface_lines(self):
        geo_lines = []
        for i in range(len(self.geo_points) - 1):
            geo_lines.append(
                self.geometry.add_line(self.geo_points[i], self.geo_points[i + 1])
            )
        # Do I need this last connection?
        geo_lines.append(
            self.geometry.add_line(self.geo_points[-1], self.geo_points[0])
        )

        return geo_lines
