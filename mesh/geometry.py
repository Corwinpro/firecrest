import os
import pygmsh as pg
import meshio
import numpy as np
import subprocess
from dolfin_utils.meshconvert import meshconvert
import dolfin as dolf

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


class Geometry:
    def __init__(self, dimensions=2):
        newfolder("Mesh")
        self.mesh_name = "Mesh/mesh"
        self.dimensions = dimensions
        self.geo_file = self.mesh_name + GEO_EXT
        self.msh_file = self.mesh_name + MSH_EXT
        self.log_file = self.mesh_name + LOG_EXT
        self.dolf_file = self.mesh_name + "." + MSH_FORMAT

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
        with open(self.log_file, "w") as mesh_log:
            mesh_log.write(gmsh_out)
            mesh_log.write(gmsh_err)

    def compile_mesh(self, mesh_format=MSH_FORMAT):
        """
        Full mesh compilation routine.
        First, creates a .msh file from .geo using self.geo_to_mesh() method,
        Second, creates an .xdmf or .xml dolfin-readable file self.dolf_file
        """
        msh_file = self.geo_to_mesh()

        if not msh_file:
            return False

        if self.dimensions == 3 and mesh_format == "xdmf":
            mesh_format = "xml"

        if mesh_format == "xml":
            try:
                meshconvert.convert2xml(self.msh_file, self.dolf_file)
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
