function [mauc] = VDBC(trainSet, trainTargets, testSet, testTargets, numNeighbor, numDim)
% VDBC algorithm with modification. During training phase there will be
% considered k-NN instances. For each instance its k nearest neighbors may
% be composed by centroids and other instances. For simplification purposes
% both sets are merged
    
    centroids = [];
    
    centroids = train(trainSet, trainTargets, numDim, centroids, numNeighbor);
    output = testing(testSet, centroids);
    % size(unique(testTargets), 1) ---> sometimes not all classes are
    % present on test set, so it is passed the size of present classes
    mauc = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
end

function centroids = train(trainSet, trainTargets, numDim, centroids, numNeighbor)
% Training function
    
    % For each training instance
    for i=1:size(trainSet, 1)
        dist = distance(transpose(trainSet(i, :)), transpose(trainSet));
        dist(i) = NaN; % distance to itself
        [nearInstD, nearInst] = sort(dist);
        
        if ~isempty(centroids)
            % If there already exist at least one centroid
            % The last column of a centroid holds its class, therefore it
            % is not counted during distance calculation
            dist2 = distance(transpose(trainSet(i, :)), transpose(centroids(:, 1:end-1)));
            [nearCentD, nearCent] = sort(dist2);
            
            
            nearestInst = [];
            nearestCent = [];
            flag = 1; % For centroids. See explanation below
            for n=1:numNeighbor
                % For instances there will always be n nearest neighbors,
                % which is not true for centroids, e.g., if only one
                % centroid was created and n == 2
                if flag <= size(nearCent, 2)
                    if nearInstD(n) < nearCentD(flag)
                        nearestInst = [nearestInst nearInst(n)];
                    else
                        nearestCent = [nearestCent nearCent(flag)];
                        flag = flag+1;
                    end
                else
                    % In here there is no more near centroids to fill the
                    % numNeighbors
                    nearestInst = [nearestInst nearInst(n)];
                end
            end
            
            % If all nearest neighbors are centroids of the same class, a
            % new centroid is created among them, then they're erased
            if size(nearestCent, 2) == numNeighbor
                possibleClasses = centroids(nearestCent, end);
                if all(possibleClasses == possibleClasses(1))
                    % A centroid of centroids
                    newCtr = zeros(1, numDim+1);
                    for d=1:numDim
                        newCtr(1, d) = mean(centroids(nearestCent, d));
                    end
                    newCtr(1, end) = possibleClasses(1);
                    centroids = [centroids; newCtr];
                    % Erasing centroids
                    centroids(nearestCent, :) = [];
                else
                    % At least one centroid is from different class
                    newCtr = zeros(1, numDim+1);
                    newCtr(1, 1:end-1) = trainSet(i, :); 
                    newCtr(1, end) = trainTargets(i);
                    centroids = [centroids; newCtr];
                end
            else
                % Nearest neighbors are composed by instances and centroids
                % or only by instances
                possibleClasses = zeros(1, numNeighbor);
                possibleClasses(1, 1:size(nearestInst, 2)) = trainSet(nearestInst);
                possibleClasses(1, size(nearestInst, 2)+1:end) = centroids(nearestCent, end);
                if all(possibleClasses == possibleClasses(1))
                    % If all belong to the same class, create a centroid 
                    % among them
                    newCtr = zeros(1, numDim+1);
                    for d=1:numDim
                        newCtr(1, d) = mean([trainSet([i nearestInst], d); centroids(nearestCent, d)]);
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
            end
        else
            % Centroids set is still empty
            nearest = nearInst(numNeighbor);
            possibleClasses = zeros(1, size(nearest,2));
            for j=1:size(nearest,2)
                possibleClasses(j) = trainTargets(nearest(j));
            end
            
            if all(possibleClasses == possibleClasses(1))
                % If all belong to the same class, create a centroid among 
                % them
                newCtr = zeros(1, numDim+1);
                for d=1:numDim
                    newCtr(1, d) = mean(trainSet([i nearest], d));
                end
                newCtr(1, end) = possibleClasses(1);
                centroids = [centroids; newCtr];
            else
                % If at least one of them belongs to different class, the 
                % current instance becomes a centroid
                newCtr = zeros(1, numDim+1);
                newCtr(1, 1:end-1) = trainSet(i, :); 
                newCtr(1, end) = trainTargets(i);
                centroids = [centroids; newCtr];
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
