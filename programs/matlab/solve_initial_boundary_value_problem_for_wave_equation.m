end_time = 0.5;


model = createpde();
geometryFromEdges(model,@circleg);
specifyCoefficients(model,'m',1,'d',0,'c',1,'a',0,'f',0);
generateMesh(model);
setInitialConditions(model,0,0);
applyBoundaryCondition(model,'neumann', ...
                             'Edge',1:model.Geometry.NumEdges, ...
                             'g', @neuman_control);

solution = solvepde(model,[0 end_time]);

pdeplot(model,'XYData',solution.NodalSolution(:,2))


function value = neuman_control(location,state)
    value = sin(atan2(location.y,location.x));
end
