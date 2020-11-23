import json
import decimal
from decimal import Decimal
import logging

from firecrest.models.geometry_registry.geometry_registry import geometry_registry
from firecrest.models.free_surface_cap import SurfaceModelFactory
from firecrest.models.inflow import InflowModelFactory
from firecrest.misc.logger import LoggerFactory
from firecrest.solvers.unsteady_optimizer import UnsteadyOptimizationSolver
from firecrest.misc.time_storage import ControlSpaceFactory

decimal.getcontext().prec = 6
log = logging.getLogger(__name__)


class FirecrestApp:
    def __init__(self, workflow_file, run_mode=None):
        self.workflow_file = workflow_file
        with open(self.workflow_file) as f:
            self.setup_data = json.load(f)

        self.run_mode = run_mode or self.setup_data["mode"] or "optimization"
        self.setup_data["mode"] = self.run_mode

        self.timescale = Decimal("0.1")  # in microseconds

        self.time_domain_configuration = self._configure_time_domain()
        self.control_space_configuration = self._configure_control_space()
        self.experiment_id = self._configure_experiment_id()
        self.log_config = self._configure_logging()
        self.constants = self._configure_constants()
        self.domain_configuration = self._configure_domain()

        self.factories = self.setup_model_factories()

    def setup_model_factories(self):
        factories = {
            "models": {
                "surface_model": SurfaceModelFactory(self.setup_data),
                "inflow_model": InflowModelFactory(),
            },
            "solvers": {"unsteady_solver": None},
            "loggers": {
                "base_logger": LoggerFactory(
                    run_mode=self.run_mode,
                    experiment_id=self.experiment_id,
                    log_names=self.log_config,
                )
            },
            "control_spaces": {
                "piecewise_linear": ControlSpaceFactory(
                    dt=self.time_domain_configuration["dt"],
                    final_time=self.time_domain_configuration["T"],
                    signal_window=self.control_space_configuration["window"],
                    reduced_basis=True,
                ).create_piecewise_linear_space()
            },
        }
        return factories

    def run(self):
        log.info(
            f"Started {__name__} application run.\n"
            f"Workflow File: {self.workflow_file}\n"
            f"Run mode: {self.run_mode}\n"
            f"Experiment ID: {self.experiment_id}\n"
        )

        domain = self.domain_configuration["domain"]
        logger = self.factories["loggers"]["base_logger"].create_basic_logger()
        model_factory = self.factories["models"]
        optimizer = UnsteadyOptimizationSolver(
            domain,
            Re=self.constants["Re"],
            Pr=self.constants["Pr"],
            timer=self.time_domain_configuration,
            experiment_id=self.experiment_id,
            initial_state=(0.0, (0.0, 0.0), 0.0),
            control_boundary=self.domain_configuration["control_boundary"],
            shared_boundary=self.domain_configuration["shared_boundary"],
            logger=logger,
            model_factory=model_factory,
        )
        result = optimizer.run(
            self.control_space_configuration["control_default"],
            self.control_space_configuration["bounds"],
            signal_window=self.control_space_configuration["window"],
            optimization_method=self.control_space_configuration["algorithm"],
        )

    def _configure_domain(self):
        # Geometry data for acoustic domain
        geometry_data = self.setup_data["acoustic_domain"]
        geometry_type = geometry_data.get("type", "symmetric_printhead")
        geometry_assembler = geometry_registry[geometry_type](geometry_data)
        return {
            "domain": geometry_assembler.domain,
            "control_boundary": geometry_assembler.control_boundary,
            "shared_boundary": geometry_assembler.shared_boundary,
        }

    def _configure_constants(self):
        # Constants parsing
        constants_data = self.setup_data["constants"].copy()
        constants_data["Re"] = (
            constants_data["density"]
            * constants_data["sound_speed"]
            * constants_data["length"]
            / constants_data["viscosity"]
        )
        # L = constants_data["length"]
        # c_s = constants_data["sound_speed"]
        # rho = constants_data["density"]
        # epsilon = constants_data["Mach"]
        # gamma_st = constants_data["surface_tension"]
        # mu = constants_data["viscosity"]
        # Pr = constants_data["Pr"]
        # Re = rho * c_s * L / mu
        return constants_data

    def _configure_time_domain(self):
        # Time domain data
        time_domain_data = self.setup_data["time_domain"]
        final_time = Decimal(str(time_domain_data["final_time"]))  # in microseconds
        time_step = Decimal(str(time_domain_data["dt"]))  # in microseconds
        nondim_final_time = final_time / self.timescale
        nondim_time_step = time_step / self.timescale
        timer = {"dt": nondim_time_step, "T": nondim_final_time}
        return timer

    def _configure_control_space(self):
        # Control data
        waveform_data = self.setup_data["control_space"]
        control_type = waveform_data["type"]
        waveform_window = waveform_data["window"]  # in microseconds
        nondim_waveform_window = waveform_window / float(self.timescale)
        control_default_value = waveform_data.get("control_default", None)
        control_upper_limit = waveform_data.get("upper_limit", 0.015)
        control_lower_limit = waveform_data.get("lower_limit", -0.015)
        control_algorithm = waveform_data.get("algorithm", "L-BFGS-B")

        return {
            "control_type": control_type,
            "algorithm": control_algorithm,
            "control_default": control_default_value,
            "window": nondim_waveform_window,
            "bounds": (control_lower_limit, control_upper_limit),
        }

    def _configure_logging(self):
        # Logging settings
        logging_data = self.setup_data["logging"]
        plot_every = logging_data["plot_frequency"]
        optimization = logging_data.get("optimisation", "optimisation_log_")
        in_process = logging_data.get("energy_history", "energy_history_")
        return {
            "plot_every": plot_every,
            "optimization": optimization,
            "in_process": in_process,
        }

    def _configure_experiment_id(self):
        experiment_id = (
            self.setup_data["acoustic_domain"]["type"]
            + "_act_"
            + str(self.setup_data["acoustic_domain"]["actuator_length"])
            + "_window_"
            + str(self.setup_data["control_space"]["window"])
            + "_T_"
            + str(self.setup_data["time_domain"]["final_time"])
        )
        return experiment_id
