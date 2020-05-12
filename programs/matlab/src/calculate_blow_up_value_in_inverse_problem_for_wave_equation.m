END_TIME = 1;
SPACE_BASIS_FUNCTION_COUNT = 1;
TARGET_POINT = [0 0];
TIME_BASIS_FUNCTION_COUNT = 8;

global TIME_STEP_SIZE

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

applyBoundaryCondition(model,'neumann', ...
                             'Edge',1:model.Geometry.NumEdges, ...
                             'g', @neumanControl);

global iSpaceBasisFunction
global jtimeBasisFunction

for iSpaceBasisFunction = 0:SPACE_BASIS_FUNCTION_COUNT - 1
    for jtimeBasisFunction = 1:TIME_BASIS_FUNCTION_COUNT
        solution = solvepde(model,[0 END_TIME]);
        solutions(:, ...
            iSpaceBasisFunction*SPACE_BASIS_FUNCTION_COUNT ...
            + jtimeBasisFunction) ...
            = solution.NodalSolution(:,2);
    end
end

applyBoundaryCondition(model,'neumann', ...
                             'Edge',1:model.Geometry.NumEdges, ...
                             'g', 0);
finiteElementMatrices = assembleFEMatrices(model);

connectionMatrix = transpose(solutions)*finiteElementMatrices.M*solutions;

borderControlBasisCoefficients = ...
    pinv(connectionMatrix) ...
    *transpose(solutions) ...
    *finiteElementMatrices.M ...
    *ones(numberOfNodes, 1);

oneWave = solutions*borderControlBasisCoefficients;

disp(['Blow up integral: ' ...
    num2str(sum(transpose(oneWave) ...
    .*log((model.Mesh.Nodes(1,:)-TARGET_POINT(1)).^2 ...
    +(model.Mesh.Nodes(2,:)-TARGET_POINT(2)).^2)) ...
    /numel(model.Mesh.Nodes) ...
    *pi)])


function result = neumanControl(location,state)
    global TIME_STEP_SIZE
    global iSpaceBasisFunction
    global jtimeBasisFunction

    if isnan(state.time)
        result = NaN(size(location.x));
        return
    end

    if state.time<(jtimeBasisFunction-1)*TIME_STEP_SIZE ...
            ||state.time>jtimeBasisFunction*TIME_STEP_SIZE
        result = 0;
        return
    end

    if iSpaceBasisFunction==0
        result = 1;
        return
    end

    cosineAndSineCellArray = {@(x) cos(x) @(x) sin(x)};
    currentSpaceBasisFunction = ...
        cosineAndSineCellArray{1+mod(iSpaceBasisFunction,2)};
    currentSpaceBasisFunctionArgument = ...
        ceil(iSpaceBasisFunction/2)*atan2(location.y,location.x);

    result = currentSpaceBasisFunction(currentSpaceBasisFunctionArgument);

    return
end
