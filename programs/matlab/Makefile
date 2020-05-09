SOURCE_FILENAME := solve_initial_boundary_value_problem_for_wave_equation.m


check: lint

lint:
	matlab -nojvm -r "mlint('$(SOURCE_FILENAME)'); exit"

run:
	DISPLAY=:0 matlab -nodesktop -r "run('$(SOURCE_FILENAME)')"

.PHONY: check lint run
