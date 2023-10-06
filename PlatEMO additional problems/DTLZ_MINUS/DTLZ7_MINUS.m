classdef DTLZ7_MINUS < PROBLEM
% <multi/many> <real> <large/none> <expensive/none>
% Minus-DTLZ7

%------------------------------- Reference --------------------------------
% H. Ishibuchi, Y. Setoguchi, H. Masuda, and Y. Nojima, Performance of
% Decomposition-Based Many-Objective Algorithms Strongly Depends on Pareto
% Front Shapes, IEEE Transactions on Evolutionary Computation, 2017, 21(2):
% 169-190.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 3; end
            if isempty(obj.D); obj.D = obj.M+19; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopObj = zeros(size(PopDec,1),obj.M);
            g      = 1+9*mean(PopDec(:,obj.M:end),2);
            PopObj(:,1:obj.M-1) = PopDec(:,1:obj.M-1);
            PopObj(:,obj.M)     = (1+g).*(obj.M-sum(PopObj(:,1:obj.M-1)./(1+repmat(g,1,obj.M-1)).*(1+sin(3*pi.*PopObj(:,1:obj.M-1))),2));
            PopObj = -PopObj;
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R = [];
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = [];
        end
    end
end