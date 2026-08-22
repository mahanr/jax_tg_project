class MhdRK4Integrator:
    def __init__(self, operator, dt, viscosity, resistivity):
        self.operator = operator
        self.dt = dt
        self.viscosity = viscosity
        self.resistivity = resistivity

    def _add_state_rhs(self, velocity, magnetic_field, scale, dudt, dbdt):
        return (
            velocity + scale * dudt,
            magnetic_field + scale * dbdt,
        )

    def step(self, state):
        velocity, magnetic_field = state
        k1_u, k1_b = self.operator.rhs(
            velocity, magnetic_field, self.viscosity, self.resistivity
        )
        u2, b2 = self._add_state_rhs(velocity, magnetic_field, 0.5 * self.dt, k1_u, k1_b)
        k2_u, k2_b = self.operator.rhs(u2, b2, self.viscosity, self.resistivity)
        u3, b3 = self._add_state_rhs(velocity, magnetic_field, 0.5 * self.dt, k2_u, k2_b)
        k3_u, k3_b = self.operator.rhs(u3, b3, self.viscosity, self.resistivity)
        u4, b4 = self._add_state_rhs(velocity, magnetic_field, self.dt, k3_u, k3_b)
        k4_u, k4_b = self.operator.rhs(u4, b4, self.viscosity, self.resistivity)

        velocity_new = velocity + (self.dt / 6.0) * (
            k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u
        )
        magnetic_field_new = magnetic_field + (self.dt / 6.0) * (
            k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b
        )
        return velocity_new, magnetic_field_new
