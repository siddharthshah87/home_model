# models/panel_model.py

class SmartPanelConfig:
    def __init__(self,
                 panel_type="Legacy",
                 control_ev=False,
                 control_hvac=False,
                 control_water_heater=False,
                 control_solar_inverter=False,
                 control_battery=False):
        self.panel_type = panel_type  # "Legacy", "Smart Subpanel", "Smart Full Panel"
        self.control_ev = control_ev
        self.control_hvac = control_hvac
        self.control_water_heater = control_water_heater
        self.control_solar_inverter = control_solar_inverter
        self.control_battery = control_battery

    def is_load_controlled(self, load_type):
        """
        load_type: one of ["ev", "hvac", "water_heater", "solar_inverter", "battery"]
        """
        if self.panel_type == "Legacy":
            return False
        if load_type == "ev":
            return self.control_ev
        if load_type == "hvac":
            return self.control_hvac
        if load_type == "water_heater":
            return self.control_water_heater
        if load_type == "solar_inverter":
            return self.control_solar_inverter
        if load_type == "battery":
            return self.control_battery
        return False
