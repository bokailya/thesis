END_TIME = 0.5;
SPACE_BASIS_FUNCTION_COUNT = 2;
TIME_BASIS_FUNCTION_COUNT = 2;

global TIME_STEP_SIZE

TIME_STEP_SIZE = END_TIME / TIME_BASIS_FUNCTION_COUNT;


model = createpde();
geometryFromEdges(model,@circleg);
specifyCoefficients(model,'m',1,'d',0,'c',1,'a',0,'f',0);
generateMesh(model);
setInitialConditions(model,0,0);


global iSpaceBasisFunction
global jtimeBasisFunction


sizeOfNodes = size(model.Mesh.Nodes);
solutions = ...
    zeros(sizeOfNodes(2), ...
    SPACE_BASIS_FUNCTION_COUNT * TIME_BASIS_FUNCTION_COUNT);

for iSpaceBasisFunction = 0:SPACE_BASIS_FUNCTION_COUNT - 1
    for jtimeBasisFunction = 1:TIME_BASIS_FUNCTION_COUNT
        applyBoundaryCondition(model,'neumann', ...
                                     'Edge',1:model.Geometry.NumEdges, ...
                                     'g', @neumanControl);

        solution = solvepde(model,[0 END_TIME]);
        solutions(:, ...
            iSpaceBasisFunction * SPACE_BASIS_FUNCTION_COUNT ...
            + jtimeBasisFunction) ...
            = solution.NodalSolution(:,2);
    end
end

for iSolution = 1:SPACE_BASIS_FUNCTION_COUNT * TIME_BASIS_FUNCTION_COUNT
    subplot(SPACE_BASIS_FUNCTION_COUNT, TIME_BASIS_FUNCTION_COUNT, iSolution)
    pdeplot(model,'XYData',solutions(:,iSolution))
end

[~,finiteElementMassMatrix,~] = assema(model,1,0,0);
connectionMatrix = transpose(solutions) * finiteElementMassMatrix * solutions;


function result = neumanControl(location,state)
    global TIME_STEP_SIZE
    global iSpaceBasisFunction
    global jtimeBasisFunction

    if isnan(state.time)
        result = NaN(size(location.x));
        return
    end

    if ...
            state.time < (jtimeBasisFunction - 1) * TIME_STEP_SIZE ...
            || state.time > jtimeBasisFunction * TIME_STEP_SIZE
        result = 0;
        return
    end

    if iSpaceBasisFunction == 0
        result = 1;
        return
    end

    cosineAndSineCellArray = {@(x) cos(x) @(x) sin(x)};
    currentSpaceBasisFunction = ...
        cosineAndSineCellArray{1 + mod(iSpaceBasisFunction, 2)};
    currentSpaceBasisFunctionArgument = ...
        ceil(iSpaceBasisFunction / 2) * atan2(location.y, location.x);

    result = currentSpaceBasisFunction(currentSpaceBasisFunctionArgument);

    return
end
