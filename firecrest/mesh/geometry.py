from collections import defaultdict
import logging
import os
import meshio
from abc import ABC
import dolfin as dolf
import gmsh

logger = logging.getLogger(__name__)


class _Point:
    def __init__(self, x, y, z=0.0, mesh_size=None):
        self.coordinates = (x, y, z)
        self.mesh_size = mesh_size

    def __repr__(self) -> str:
        return f"_Point({self.coordinates!r}, {self.mesh_size})"


class MeshBuilder:
    _model_id = 1

    def __init__(self, boundary_elements, name="geometry"):
        self._model_id = MeshBuilder._model_id
        MeshBuilder._model_id += 1

        self.boundary_elements = boundary_elements
        mesh = self.generate_meshio_mesh(mesh_name=name)
        self.xdmf_mesh, (self.facet_xdmf_mesh, self.physical_label) = self.write_xdmf(
            from_mesh=mesh, output_name=name
        )

    def _get_points(self, points, mesh_size):
        points = [_Point(*point, mesh_size=mesh_size) for point in points]
        return points

    def generate_meshio_mesh(self, mesh_name, verbose=False) -> meshio.Mesh:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(f"Gmsh model {self._model_id}")

        points_of_boundaries = [
            self._get_points(boundary_element.surface_points, boundary_element.el_size)
            for boundary_element in self.boundary_elements
        ]
        self._merge_end_points(points_of_boundaries)

        points_collection = [
            self.add_gmsh_points(points_of_boundary)
            for points_of_boundary in points_of_boundaries
        ]

        for i in range(len(points_collection) - 1):
            points_collection[i].append(points_collection[i+1][0])
        points_collection[-1].append(points_collection[0][0])

        lines_collection = [
            self.add_gmsh_lines(points_set) for points_set in points_collection
        ]
        all_lines = sum(lines_collection, [])

        lineloop = gmsh.model.geo.add_curve_loop(all_lines, 1)
        surface_tag = gmsh.model.geo.add_plane_surface([lineloop], 1)

        gmsh.model.geo.synchronize()
        for boundary_element, boundary_lines in zip(self.boundary_elements, lines_collection):
            index = boundary_element.surface_index
            tag = gmsh.model.addPhysicalGroup(1, boundary_lines, index)
            gmsh.model.setPhysicalName(1, tag, f"Boundary{index}")

        surface_physical_group = gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, surface_physical_group, "Surface")

        gmsh.model.mesh.generate(2)

        msh_file_name = f"{mesh_name}.msh"
        gmsh.write(msh_file_name)
        gmsh.finalize()

        meshio_mesh = meshio.read(msh_file_name)
        return meshio_mesh

    @staticmethod
    def add_gmsh_points(surface_points: list[_Point]) -> list[int]:
        points = []
        for surface_point in surface_points:
            tag = gmsh.model.geo.add_point(
                *surface_point.coordinates, surface_point.mesh_size
            )
            points.append(tag)
        return points

    @staticmethod
    def _merge_end_points(points_collection: list[list[_Point]]):
        for i, _ in enumerate(points_collection):
            previous_point_set = points_collection[i-1]
            current_point_set = points_collection[i]

            last_point = previous_point_set[-1]
            first_point = current_point_set[0]

            min_mesh_size = min(last_point.mesh_size, first_point.mesh_size)
            last_point.mesh_size = first_point.mesh_size = min_mesh_size
            points_collection[i-1] = points_collection[i-1][:-1]

    @staticmethod
    def add_gmsh_lines(points_set):
        lines = []
        for i in range(len(points_set) - 1):
            tag = gmsh.model.geo.addLine(points_set[i], points_set[i+1])
            lines.append(tag)
        return lines

    def write_xdmf(
        self, from_mesh: meshio.Mesh, output_name: str, physical_label: str = "physical"
    ) -> tuple[str, tuple[str, str]]:

        def create_mesh(mesh_, cell_type, prune_z=True):
            cells = mesh_.get_cells_type(cell_type)
            physical_data = mesh_.get_cell_data("gmsh:physical", cell_type)
            out_mesh = meshio.Mesh(
                points=mesh_.points,
                cells={cell_type: cells},
                cell_data={physical_label: [physical_data]}
            )
            if prune_z:
                out_mesh.prune_z_0()
            return out_mesh

        line_mesh = create_mesh(from_mesh, "line")
        facet_mesh_name = f"{output_name}_facet.xdmf"
        meshio.write(facet_mesh_name, line_mesh)

        triangle_mesh = create_mesh(from_mesh, "triangle")
        mesh_name = f"{output_name}.xdmf"
        meshio.write(mesh_name, triangle_mesh)

        return mesh_name, (facet_mesh_name, physical_label)


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

    attributes:
    - boundary_elements: collection of boundary_elements forming the Geometry
    """

    def __init__(
        self,
        boundary_elements,
        mesh_name="mesh",
        mesh_folder="Mesh",
    ):
        self.boundary_elements = boundary_elements
        self.mesh_folder = mesh_folder
        self.mesh_name = mesh_name

        if self.mesh_folder:
            newfolder(self.mesh_folder)
            self.mesh_name = self.mesh_folder + "/" + mesh_name

        self.mark_boundaries()

        self.mesh = dolf.Mesh()
        self._boundary_parts = None

    def mark_boundaries(self):
        """
        The dictionary 'markers_dict' stores the pairs of
        (keys) boundary condition type,
        (values) list of boundary_elements of this boundary type,
        i.e.
        ```
        self.market_dict = {
        "noslip": [noslip_boundary_1, noslip_boundary_2],
        "adiabatic": [noslip_boundary_1],
        "heat_flux": [noslip_boundary_2]
        }
        ```

        """
        markers_dict = defaultdict(list)
        for boundary_element in self.boundary_elements:
            for btype in boundary_element.bcond:
                markers_dict[btype].append(boundary_element)
        return markers_dict

    @property
    def boundary_parts(self):
        """
        Creates dolfin MeshFunction of dim-1 dimension
        """
        raise NotImplementedError

    @property
    def ds(self):
        """
        Creates dolfin 'ds' measure, accounting for physical marking
        """
        return dolf.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_parts)

    @property
    def dx(self):
        """
        Creates dolfin 'dx' measure
        """
        return dolf.dx(domain=self.mesh)

    @property
    def n(self):
        """
        Creates dolfin mesh normal
        """
        return dolf.FacetNormal(self.mesh)

    def _get_boundaries(self, boundary_type):
        """
        Returns all boundaries of the given boundary_type: str
        """
        return self.markers_dict[boundary_type]

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
    mesh_reader = dolf.XDMFFile

    def __init__(
        self,
        boundary_elements,
        mesh_name="mesh",
        mesh_folder="Mesh",
        refinement_level=0,
    ):
        super().__init__(boundary_elements, mesh_name, mesh_folder)
        mesh_builder = MeshBuilder(
            boundary_elements=boundary_elements, name=self.mesh_name
        )

        with self.mesh_reader(mesh_builder.xdmf_mesh) as file:
            file.read(self.mesh)

        self.facet_xdmf_mesh = mesh_builder.facet_xdmf_mesh
        self.physical_label = mesh_builder.physical_label

        if refinement_level > 0:
            dolf.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
            for _ in range(refinement_level):
                cf = dolf.MeshFunction("bool", self.mesh, True)
                cf.set_all(True)
                self.mesh = dolf.refine(self.mesh, cf)
                self._boundary_parts = dolf.adapt(self.boundary_parts, self.mesh)
            logging.info(
                f"Refinement level: {refinement_level}, "
                f"nof cells: {len(list(dolf.cells(self.mesh)))}"
            )

    @property
    def boundary_parts(self):
        """
        Creates dolfin MeshFunction of dim-1 dimension
        """
        if self._boundary_parts:
            return self._boundary_parts

        mvc = dolf.MeshValueCollection("size_t", self.mesh, 1)
        with self.mesh_reader(self.facet_xdmf_mesh) as file:
            file.read(mvc, self.physical_label)

        self._boundary_parts = dolf.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        return self._boundary_parts
