SOURCE_FILENAME := \
	src/solve_initial_boundary_value_problems_with_basis_controls.m


check: lint

lint:
	matlab -nojvm -r "mlint('$(SOURCE_FILENAME)'); exit"

run:
	DISPLAY=:0 matlab -nodesktop -r "run('$(SOURCE_FILENAME)')"

.PHONY: check lint run
