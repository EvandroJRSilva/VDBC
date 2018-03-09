function [mauc] = VDBC(trainSet, trainTargets, testSet, testTargets, numDim, numCls, numFolds)
% VDBC with a fourth modification. This version is not presented in any
% paper. Difference from Original VDBC:
%   - Tomek links are found
%       + Erase from training set the instance from 'biggest' class
%           - No erase if the 'biggest' class size is <= number of folds
%
% The construction of centroids set remains as the original one, except for
% the fact that now instances are selected randomly and then discarded

    % Finding classes sizes
    clsSize = zeros(1, numCls);
    for c=1:numCls
        clsSize(c) = size(find(trainTargets == c), 1);
    end
    
    
    [trainSet, trainTargets] = tomekLink(trainSet, trainTargets, clsSize, numFolds);
    
    % From now on the algorithm remains the same of original VDBC
    centroids = [];
    
    centroids = train(trainSet, trainTargets, numDim, centroids);
    output = testing(testSet, centroids);
    % size(unique(testTargets), 1) ---> sometimes not all classes are
    % present on test set, so it is passed the size of present classes
    mauc = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
end

function [trainSet, trainTargets] = tomekLink(trainSet, trainTargets, clsSize, numFolds)
% Finds pairs of instances that form Tomek Links and erase the instance of
% bigger class, if the bigger class has more training instances than the
% number of folds
    
    % List of instances that form Tomek Links
    tlList = [];
    for i=1:size(trainSet, 1)
        if ~ismember(i, tlList)
            dist = distance(transpose(trainSet(i, :)), transpose(trainSet));
            dist(i) = NaN; % Distance to itself
            nearest = find(dist == min(dist));
        
            if size(nearest, 2) == 1
                dist2 = distance(transpose(trainSet(nearest, :)), transpose(trainSet));
                dist2(nearest) = NaN;
                nearest2 = find(dist2 == min(dist2));
            
                if size(nearest2, 2) == 1 && nearest2 == i
                    if trainTargets(i) ~= trainTargets(nearest)
                        tlList = [tlList; nearest nearest2];
                    end
                end
            end
        end
    end
    
    if ~isempty(tlList)
        % List of instances to be erased
        toErase = [];
        for i=1:size(tlList, 1)
            cls1 = trainTargets(tlList(i, 1));
            cls2 = trainTargets(tlList(i, 2));
            
            if clsSize(cls1) > clsSize(cls2)
                if clsSize(cls1) > numFolds
                    toErase = [toErase tlList(i, 1)];
                end
            else
                if clsSize(cls2) > numFolds
                    toErase = [toErase tlList(i, 2)];
                end
            end
        end
        trainSet(toErase, :) = []; trainTargets(toErase) = [];
    end
end

function centroids = train(trainSet, trainTargets, numDim, centroids)
% Training function
    
    % Selecting instances randomly
    while ~isempty(trainSet)
        cp = randi(size(trainSet, 1));
        dist = distance(transpose(trainSet(cp, :)), transpose(trainSet));
        dist(cp) = NaN; % distance to itself
        if ~isempty(centroids)
            % If there already exist at least one centroid
            % The last column of a centroid holds its class, therefore it
            % is not counted during distance calculation
            dist2 = distance(transpose(trainSet(cp, :)), transpose(centroids(:, 1:end-1)));
            
            if min(dist) < min(dist2)
                % If another instance is closer than any centroid
                nearest = find(dist == min(dist));
                
                if size(nearest,2) > 1
                    % If two or more instances are equidistant
                    possibleClasses = zeros(1, size(nearest,2));
                    for j=1:size(nearest,2)
                        possibleClasses(j) = trainTargets(nearest(j));
                    end
                    
                    if all(possibleClasses == possibleClasses(1))
                        % If all belong to the same class, create a
                        % centroid among them
                        newCtr = zeros(1, numDim+1);
                        for d=1:numDim
                            newCtr(1, d) = mean(trainSet([cp nearest], d));
                        end
                        newCtr(1, end) = possibleClasses(1);
                        centroids = [centroids; newCtr];
                    else
                        % If at least one of them belongs to different
                        % class, the current instance becomes a centroid
                        newCtr = zeros(1, numDim+1);
                        newCtr(1, 1:end-1) = trainSet(cp, :); 
                        newCtr(1, end) = trainTargets(cp);
                        centroids = [centroids; newCtr];
                    end
                else
                    % Only one nearest neighbor
                    if trainTargets(cp) == trainTargets(nearest)
                        % If they belong to the same class, a centroid is
                        % created between them
                        newCtr = zeros(1, numDim+1);
                        for d=1:numDim
                            newCtr(d) = mean(trainSet([cp nearest], d));
                        end
                        newCtr(1, end) = trainTargets(cp);
                        centroids = [centroids; newCtr];
                    else
                        % If they belong to different classes the current
                        % instance becomes a centroid
                        newCtr = zeros(1, numDim+1);
                        newCtr(1, 1:end-1) = trainSet(cp, :); 
                        newCtr(1, end) = trainTargets(cp);
                        centroids = [centroids; newCtr];
                    end
                end
            else
                % If a centroid is closer than any instance
                nearCtrIdx = find(dist2 == min(dist2));
                nearCtrCls = centroids(nearCtrIdx, end);
                % If they belong to the same class nothing is done.
                % Otherwise, current instance becomes a centroid
                if trainTargets(cp) ~= nearCtrCls
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(cp, :); 
                    newCtr(1, end) = trainTargets(cp);
                    centroids = [centroids; newCtr];
                end
            end
        else
            % Centroids set is still empty
            nearest = find(dist == min(dist));
            if size(nearest,2) > 1
                % If two or more instances are equidistant
                possibleClasses = zeros(1, size(nearest,2));
                for j=1:size(nearest,2)
                    possibleClasses(j) = trainTargets(nearest(j));
                end
                    
                if all(possibleClasses == possibleClasses(1))
                    % If all belong to the same class, create a centroid 
                    % among them
                    newCtr = zeros(1, numDim+1);
                    for d=1:numDim
                        newCtr(1, d) = mean(trainSet([cp nearest], d));
                    end
                    newCtr(1, end) = possibleClasses(1);
                    centroids = [centroids; newCtr];
                else
                    % If at least one of them belongs to different class, 
                    % the current instance becomes a centroid
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(cp, :); 
                    newCtr(1, end) = trainTargets(cp);
                    centroids = [centroids; newCtr];
                end
            else
                % Only one nearest neighbor
                if trainTargets(cp) == trainTargets(nearest)
                    % If they belong to the same class, a centroid is
                    % created between them
                    newCtr = zeros(1, numDim+1);
                    for d=1:numDim
                        newCtr(d) = mean(trainSet([cp nearest], d));
                    end
                    newCtr(1, end) = trainTargets(cp);
                    centroids = [centroids; newCtr];
                else
                    % If they belong to different classes the current
                    % instance becomes a centroid
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(cp, :); 
                    newCtr(1, end) = trainTargets(cp);
                    centroids = [centroids; newCtr];
                end
            end  
        end
        trainSet(cp, :) = []; trainTargets(cp) = [];
    end
end

function output = testing(testSet, centroids)
% Function for generating the output based on 1NN classification with
% prototypes
    
    output = zeros(size(testSet, 1), 1);
    for i=1:size(testSet, 1)
        dist = distance(transpose(testSet(i, :)), transpose(centroids(:, 1:end-1)));
        nearest = find(dist == min(dist));
        if size(nearest, 2) > 1 % The size is related to the column due to the distance function
            possibleClasses = centroids(nearest, end);
            if all(possibleClasses == possibleClasses(1))
                output(i) = possibleClasses(1);
            else
                % For while the class is decided randomly. TODO: decide it
                % based on probabilities, giving more importante to minor
                % classes
                possibleClasses = sort(possibleClasses);
                output(i) = possibleClasses(randi(size(possibleClasses, 1)));
            end
        else
            output(i) = centroids(nearest, end);
        end
    end
end

function mauc = calculateMAUC(output, testTargets, nClass)
% Function for calculating MAUC performance
    [mauc, ~] = colAUC(output, testTargets, 'ROC');
    mauc = (2/(nClass*(nClass - 1)))*sum(mauc);
end