class RK4Integrator:
    def __init__(self, operator, dt, viscosity):
        self.operator = operator
        self.dt = dt
        self.viscosity = viscosity

    def step(self, state):
        rhs_1 = self.operator.rhs(state, self.viscosity)
        rhs_2 = self.operator.rhs(state + 0.5 * self.dt * rhs_1, self.viscosity)
        rhs_3 = self.operator.rhs(state + 0.5 * self.dt * rhs_2, self.viscosity)
        rhs_4 = self.operator.rhs(state + self.dt * rhs_3, self.viscosity)
        return state + (self.dt / 6.0) * (rhs_1 + 2.0 * rhs_2 + 2.0 * rhs_3 + rhs_4)
