import json
import decimal
from decimal import Decimal
import logging

from firecrest.models.geometry_registry.geometry_registry import geometry_registry
from firecrest.misc.time_storage import TimeSeries, PiecewiseLinearBasis


decimal.getcontext().prec = 6
log = logging.getLogger(__name__)


class FirecrestApp:
    def __init__(self, workflow_file, run_mode=None):
        self.workflow_file = workflow_file
        with open(self.workflow_file) as f:
            self.setup_data = json.load(f)

        self.run_mode = run_mode or "optimization"
        if run_mode is not None:
            self.setup_data["mode"] = run_mode

        self.timescale = Decimal("0.1")  # in microseconds

        # self.domain_configuration = self._configure_domain()
        self.time_domain_configuration = self._configure_time_domain()
        self.control_space_configuration = self._configure_control_space()
        self.experiment_id = self._configure_experiment_id()

    def run(self):
        log.info(
            f"Started {__name__} application run.\n"
            f"Workflow File: {self.workflow_file}\n"
            f"Run mode: {self.run_mode}\n"
            f"Experiment ID: {self.experiment_id}\n"
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

    def _configure_dim_constants(self):
        # Dimensional constants parse
        constants_data = self.setup_data["constants"]
        L = constants_data["length"]
        c_s = constants_data["sound_speed"]
        rho = constants_data["density"]
        epsilon = constants_data["Mach"]
        gamma_st = constants_data["surface_tension"]
        mu = constants_data["viscosity"]
        Pr = constants_data["Pr"]
        Re = rho * c_s * L / mu

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
        # Waveform control data
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
        optimization_log_filename = logging_data.get(
            "optimisation", "optimisation_log_"
        )
        energy_history_log_filename = logging_data.get(
            "energy_history", "energy_history_"
        )

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

    def _configure_control_space(self):
        pass
