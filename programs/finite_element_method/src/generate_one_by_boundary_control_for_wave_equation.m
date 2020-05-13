END_TIME = 1;
SPACE_BASIS_FUNCTION_COUNT = 1;
TIME_BASIS_FUNCTION_COUNT = 8;


TIME_STEP_SIZE = END_TIME/TIME_BASIS_FUNCTION_COUNT;


model = createpde();
geometryFromEdges(model,@circleg);
specifyCoefficients(model,'m',1,'d',0,'c',1,'a',0,'f',0);
generateMesh(model);
setInitialConditions(model,0,0);


sizeOfNodes = size(model.Mesh.Nodes);
numberOfNodes = sizeOfNodes(2);

solutions = ...
    zeros(numberOfNodes, ...
    SPACE_BASIS_FUNCTION_COUNT*TIME_BASIS_FUNCTION_COUNT);

for iSpaceBasisFunction = 0:SPACE_BASIS_FUNCTION_COUNT - 1
    for jtimeBasisFunction = 1:TIME_BASIS_FUNCTION_COUNT
        currentBasisControl = ...
            @(location,state) ...
            neumanControl(location,iSpaceBasisFunction, ...
            state,jtimeBasisFunction,TIME_STEP_SIZE);

        applyBoundaryCondition(model,'neumann', ...
                                     'Edge',1:model.Geometry.NumEdges, ...
                                     'g',currentBasisControl);

        solution = solvepde(model,[0 END_TIME]);
        solutions(:, ...
            iSpaceBasisFunction * SPACE_BASIS_FUNCTION_COUNT ...
            + jtimeBasisFunction) ...
            = solution.NodalSolution(:,2);
    end
end

applyBoundaryCondition(model,'neumann', ...
                             'Edge',1:model.Geometry.NumEdges, ...
                             'g', 0);
finiteElementMatrices = assembleFEMatrices(model);

connectionMatrix = ...
    transpose(solutions) * finiteElementMatrices.M * solutions;

borderControlBasisCoefficients = ...
    pinv(connectionMatrix) ...
    * transpose(solutions) ...
    * finiteElementMatrices.M ...
    * ones(numberOfNodes, 1);

oneWave = solutions * borderControlBasisCoefficients;

pdeplot(model,'XYData',oneWave)


function result = neumanControl(location,spaceBasisFunctionIndex, ...
        state,timeBasisFunctionIndex,timeStepSize)
    if isnan(state.time)
        result = NaN(size(location.x));
        return
    end

    if state.time<(timeBasisFunctionIndex-1)*timeStepSize ...
            ||state.time>timeBasisFunctionIndex*timeStepSize
        result = 0;
        return
    end

    if spaceBasisFunctionIndex==0
        result = 1;
        return
    end

    cosineAndSineCellArray = {@(x) cos(x) @(x) sin(x)};
    currentSpaceBasisFunction = ...
        cosineAndSineCellArray{1+mod(spaceBasisFunctionIndex,2)};
    currentSpaceBasisFunctionArgument = ...
        ceil(spaceBasisFunctionIndex/2)*atan2(location.y,location.x);

    result = currentSpaceBasisFunction(currentSpaceBasisFunctionArgument);

    return
end
