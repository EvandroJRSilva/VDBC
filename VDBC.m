function [results] = VDBC(dataF, dataTNum, numCls, numDim, numFolds)
% Function VDBC for Voronoi Diagram Based Classifier. It is based on Chang
% W* algorithm:
%   - Select a random instance;
%   - Find its nearest neighbor
%       + If they belong to the same class
%           - Create a centroid between them.
%       + Else
%           - Current selected instance becomes a centroid.
%   - Test unknown instances with built centroids

%% Pre-process
    % Vector to store MAUC values
    results(numFolds) = 0;
    % Separating data into k folds
    foldIdx = kfoldIndices(numCls, dataTNum, numFolds);
    
%% Train and Test
    for k=1:numFolds
        disp(strcat('ITERAÇÃO', num2str(k)));
        
        centroids = [];
        
        % Test Set
        testIdx = foldIdx(k).indices;
        testSet = dataF(testIdx, :); testTargets = dataTNum(testIdx);
        
        % Train Set
        trainIdx = [];
        for i=1:numFolds
            if i ~= k
                trainIdx = [trainIdx; foldIdx(i).indices];
            end
        end
        trainSet = dataF(trainIdx, :); trainTargets = dataTNum(trainIdx);
        
        centroids = train(trainSet, trainTargets, numDim, centroids);
        output = testing(testSet, centroids);
        % size(unique(testTargets), 1) ---> sometimes not all classes are
        % present on test set, so it is passed the size of present classes
        results(k) = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
    end
end

function foldIdx = kfoldIndices(numCls, dataTNum, numFolds)
% Function for creating indices for the k folds    
    foldIdx(numFolds).indices = [];
    clsInd(numCls).indices = [];
    clsFoldInd(numCls).indices = [];
    % Getting the indices of instances from each class and then the inside
    % class indices for folds
    for i=1:numCls
        clsInd(i).indices = find(dataTNum == i);
        clsFoldInd(i).indices = crossvalind('Kfold', size(clsInd(i).indices, 1), numFolds);
    end
    
    for i=1:numFolds
        for j=1:numCls
            foldIdx(i).indices = [foldIdx(i).indices; clsInd(j).indices(clsFoldInd(j).indices == i)];
        end
    end
end

function centroids = train(trainSet, trainTargets, numDim, centroids)
% Training function
    
    % For each training instance
    for i=1:size(trainSet, 1)
        dist = distance(transpose(trainSet(i, :)), transpose(trainSet));
        dist(i) = NaN; % distance to itself
        if ~isempty(centroids)
            % If there already exist at least one centroid
            % The last column of a centroid holds its class, therefore it
            % is not counted during distance calculation
            dist2 = distance(transpose(trainSet(i, :)), transpose(centroids(:, 1:end-1)));
            
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
                            newCtr(1, d) = mean(trainSet([i nearest], d));
                        end
                        newCtr(1, end) = possibleClasses(1);
                        centroids = [centroids; newCtr];
                    else
                        % If at least one of them belongs to different
                        % class, the current instance becomes a centroid
                        newCtr = zeros(1, numDim+1);
                        newCtr(1, 1:end-1) = trainSet(i, :); 
                        newCtr(1, end) = trainTargets(i);
                        centroids = [centroids; newCtr];
                    end
                else
                    % Only one nearest neighbor
                    neighborCls = trainTargets(nearest);
                    
                    if trainTargets(i) == trainTargets(nearest)
                        % If they belong to the same class, a centroid is
                        % created between them
                        newCtr = zeros(1, numDim+1);
                        for d=1:numDim
                            newCtr(d) = mean(trainSet([i nearest], d));
                        end
                        newCtr(1, end) = trainTargets(i);
                        centroids = [centroids; newCtr];
                    else
                        % If they belong to different classes the current
                        % instance becomes a centroid
                        newCtr = zeros(1, numDim+1);
                        newCtr(1, 1:end-1) = trainSet(i, :); 
                        newCtr(1, end) = trainTargets(i);
                        centroids = [centroids; newCtr];
                    end
                end
            else
                % If a centroid is closer than any instance
                nearCtrIdx = find(dist2 == min(dist2));
                nearCtrCls = centroids(nearCtrIdx, end);
                % If they belong to the same class nothing is done.
                % Otherwise, current instance becomes a centroid
                if trainTargets(i) ~= nearCtrCls
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(i, :); 
                    newCtr(1, end) = trainTargets(i);
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
                        newCtr(1, d) = mean(trainSet([i nearest], d));
                    end
                    newCtr(1, end) = possibleClasses(1);
                    centroids = [centroids; newCtr];
                else
                    % If at least one of them belongs to different class, 
                    % the current instance becomes a centroid
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(i, :); 
                    newCtr(1, end) = trainTargets(i);
                    centroids = [centroids; newCtr];
                end
            else
                % Only one nearest neighbor
                neighborCls = trainTargets(nearest);
                    
                if trainTargets(i) == trainTargets(nearest)
                    % If they belong to the same class, a centroid is
                    % created between them
                    newCtr = zeros(1, numDim+1);
                    for d=1:numDim
                        newCtr(d) = mean(trainSet([i nearest], d));
                    end
                    newCtr(1, end) = trainTargets(i);
                    centroids = [centroids; newCtr];
                else
                    % If they belong to different classes the current
                    % instance becomes a centroid
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(i, :); 
                    newCtr(1, end) = trainTargets(i);
                    centroids = [centroids; newCtr];
                end
            end  
        end
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