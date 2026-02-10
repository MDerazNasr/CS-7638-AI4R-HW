######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib
    import pathlib

    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f"Unique file ID: {file_hash}")


# Asteroid motion: x(t) = c_pos + c_vel*t + 0.5*c_acc*t^2
# State: [x, vx, ax, y, vy, ay]
# Approach: keep full observation history, always use least-squares regression
# to fit the quadratic model, and predict t+1 from the fitted curve.
# Least squares is ideal here because the physics model IS a quadratic,
# so fitting a quadratic directly to the data is the optimal approach.
# More data points = better noise averaging = more accurate predictions.


class Spaceship:
    def __init__(self, bounds, xy_start):
        self.x_bounds = bounds["x"]
        self.y_bounds = bounds["y"]
        self.agent_pos_start = xy_start
        self.filters = {}
        self.agent_pos = xy_start

    def _solve_3x3(self, M, b):
        """Solve 3x3 linear system M*x = b using Cramer's rule."""

        def det3(m):
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        d = det3(M)
        if abs(d) < 1e-15:
            return None
        result = []
        for col in range(3):
            Mc = [row[:] for row in M]
            for row in range(3):
                Mc[row][col] = b[row]
            result.append(det3(Mc) / d)
        return result

    def _fit_quadratic(self, history):
        """Fit x(t) = a + b*t + c*t^2 to observation history via least squares.
        Returns (sol_x, sol_y) coefficients or None if singular.
        """
        n = len(history)
        AtA = [[0.0] * 3 for _ in range(3)]
        Atb_x = [0.0] * 3
        Atb_y = [0.0] * 3
        for i in range(n):
            t = float(i)
            row = [1.0, t, t * t]
            mx, my = history[i]
            for r in range(3):
                Atb_x[r] += row[r] * mx
                Atb_y[r] += row[r] * my
                for c in range(3):
                    AtA[r][c] += row[r] * row[c]
        sol_x = self._solve_3x3(AtA, Atb_x)
        sol_y = self._solve_3x3(AtA, Atb_y)
        if sol_x is None or sol_y is None:
            return None
        return sol_x, sol_y

    def predict_from_observations(self, asteroid_observations):
        """Predict asteroid positions at t+1."""
        predictions = {}

        # remove filters for asteroids that left the field
        gone = set(self.filters.keys()) - set(asteroid_observations.keys())
        for gid in gone:
            del self.filters[gid]

        for ast_id, (mx, my) in asteroid_observations.items():
            if ast_id not in self.filters:
                self.filters[ast_id] = {"history": [(mx, my)]}
                predictions[ast_id] = (mx, my)
                continue

            state = self.filters[ast_id]
            state["history"].append((mx, my))
            n = len(state["history"])

            if n == 2:
                # two points: linear extrapolation
                x0, y0 = state["history"][0]
                x1, y1 = state["history"][1]
                predictions[ast_id] = (x1 + (x1 - x0), y1 + (y1 - y0))

            elif n >= 3:
                # three+ points: quadratic fit
                result = self._fit_quadratic(state["history"])
                if result is not None:
                    sol_x, sol_y = result
                    t_next = float(n)
                    px = sol_x[0] + sol_x[1] * t_next + sol_x[2] * t_next * t_next
                    py = sol_y[0] + sol_y[1] * t_next + sol_y[2] * t_next * t_next
                    predictions[ast_id] = (px, py)
                else:
                    predictions[ast_id] = (mx, my)
            else:
                predictions[ast_id] = (mx, my)

        return predictions

    def jump(self, asteroid_observations, agent_data):
        """Decide which asteroid to jump onto to reach homebase (top)."""
        estimated = self.predict_from_observations(asteroid_observations)

        jump_dist = agent_data["jump_distance"]
        ridden = agent_data["ridden_asteroid"]

        # current spaceship position
        if ridden is not None and ridden in asteroid_observations:
            cx, cy = asteroid_observations[ridden]
        else:
            cx, cy = self.agent_pos
        self.agent_pos = (cx, cy)

        # already at homebase?
        y_max = self.y_bounds[1]
        if cy >= y_max:
            return None, estimated

        # use 75% of jump range as safety margin for measurement noise
        safe_jump = jump_dist * 0.75

        best_id = None
        best_score = -float("inf")
        x_lo, x_hi = self.x_bounds
        y_lo = self.y_bounds[0]

        for aid, (px, py) in estimated.items():
            if aid == ridden:
                continue
            if aid not in asteroid_observations:
                continue

            obs_x, obs_y = asteroid_observations[aid]

            # distance check using current measured position
            dx = obs_x - cx
            dy = obs_y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > safe_jump:
                continue

            # measured position must be within field
            if obs_x < x_lo or obs_x > x_hi or obs_y < y_lo or obs_y > y_max:
                continue

            # predicted position should be within field (with small margin)
            margin = 0.1
            if px < x_lo + margin or px > x_hi - margin:
                continue
            if py < y_lo + margin:
                continue

            # score: prefer highest y (closest to homebase)
            if py > best_score:
                best_score = py
                best_id = aid

        # jump if it moves us upward or we're not on any asteroid
        if best_id is not None and (best_score > cy or ridden is None):
            return best_id, estimated

        return None, estimated


def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith126).
    whoami = "mnasr34"
    return whoami
