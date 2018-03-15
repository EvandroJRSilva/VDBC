function [mauc, prototypes] = VDBC(trainSet, trainTargets, testSet, testTargets, numDim, numCls)
% This is VDBC M5, the fifth modification of VDBC. This is maybe the
% biggest modification among M1 to M5 versions. The new algorithm is as
% follows:
%   - Find the smallest distance between a pair a of instances of traning
%   set
%       + The half of this distance is how much a centroid radius will grow
%       at each time
%   - All training instances become centroid in the beggining
%   - Begenning with the smallest classes, each centroid grows its radius
%       + If the growing of a radius makes it touch or trespass the radius
%       of another centroid
%           - Merge both centroids if they are from the same class,
%           creating a new centroid between them
%           - Stop the growing of both centroids if they are from different
%           classes
%
%   Centroids are now mapped as n X d+3 matrix, in which d+1 is the label,
%   d+2 is the radius value and d+3 is a bollean indicating if the radius
%   can or cannot grow.
    
    % Growing Radius ratio
    gRadius = findGRadius(trainSet);
    
    % Setting the order of classes to grow (from lowest to highest)
    sizeCls = zeros(1, numCls);
    for c=1:numCls
        sizeCls(c) = size(find(trainTargets == c), 1);
    end
    
    [~, classOrder] = sort(sizeCls);
    
    
    % Creating and filling centroids set
    centroids = zeros(size(trainSet, 1), size(trainSet, 2)+3);
    centroids(:, 1:numDim) = trainSet; 
    centroids(:, numDim+1) = trainTargets;
    centroids(:, numDim+2) = 0; centroids(:, numDim+3) = 1;
    
    % Training
    centroids = train(centroids, gRadius, classOrder, numDim);
    prototypes = size(centroids, 1);
    % Test
    output = testing(testSet, centroids(:, 1:numDim+1));
    % size(unique(testTargets), 1) ---> sometimes not all classes are
    % present on test set, so it is passed the size of present classes
    mauc = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
end

function radius = findGRadius(trainSet)
% This function finds the smallest distance between a pair of instances and
% return half of the distance as a growing radius ratio.
    
    
    % Vector to store the distance of each instance and its nearest
    % neighbor
    smallest = zeros(1, size(trainSet, 1));
    
    % Finding the smallest distance for each instance
    for i=1:size(trainSet, 1)
        dist = distance(transpose(trainSet(i, :)), transpose(trainSet));
        dist(i) = NaN; % distance to itself is NaN
        smallest(i) = min(dist);
    end
    
    radius = min(smallest(smallest>0))/2;
end

function centroids = train(centroids, gRadius, classOrder, numDim)
% Function to update radius and number of centroids

    while any(centroids(:, end))
        for c=1:size(classOrder, 2)
            currentCls = classOrder(c);
            currentClsIdx = find(centroids(:, numDim+1) == currentCls);
            % For each centroid of the current class. The number of
            % centroids for a class is expected to change
            i = 1; %flag
            while i <= size(currentClsIdx, 1)
                % If centroid is growable
                id = currentClsIdx(i);
                if centroids(id, end)
                    % Verifying with flag. If the growth is ok the centroid
                    % is updated
                    updatingCentroid = centroids(id, :);
                    updatingCentroid(1, numDim+2) = ...
                        updatingCentroid(1, numDim+2) + gRadius;
                    % Finding its nearest neighbor(s)
                    dist = distance(transpose(centroids(id, 1:numDim)), ...
                        transpose(centroids(:, 1:numDim)));
                    dist(id) = NaN; %distance to itself
                    nearest = find(dist == min(dist));
                    if size(nearest, 2) == 1
                        % Only one nearest neighbor
                        if (dist(nearest) - updatingCentroid(1, numDim+2) - ...
                                centroids(nearest, numDim+2)) > 0
                            % If the growth of radius is possible and do
                            % not touch the radius of its nearest neighbor
                            % the current centroid is updated
                            centroids(id, :) = updatingCentroid;
                        else
                            % If the growth of radius makes it touch or
                            % trespass the other radius
                            if centroids(id, numDim+1) == centroids(nearest, numDim+1)
                                % If both centroids are from the same class
                                newCtr = zeros(1, numDim+3);
                                newCtr(1, numDim+1) = centroids(id, numDim+1);
                                newCtr(1, numDim+2) = max(centroids(id, numDim+2), ...
                                    centroids(nearest, numDim+2));
                                newCtr(1, end) = 1;
                                for d=1:numDim
                                    newCtr(1, d) = mean(centroids([id nearest], d));
                                end
                                % Inserting new centroid and erasing the
                                % other two from the set
                                centroids = [centroids; newCtr];
                                centroids([id nearest], :) = [];
                                % Updating current class indices for the
                                % while loop
                                currentClsIdx = find(centroids(:, numDim+1) == currentCls);
                            else
                                % If centroids are from different classes.
                                % The objective of the algorithm is to grow
                                % centroids to merge them. If it is not
                                % possible to grow anymore nothing is done,
                                % except changing the growth flag
                                centroids([id nearest], end) = 0;
                            end
                        end
                    else
                        % More than one (equidistant) nearest neighbor
                        
                        % Distances after radius growth
                        afterDist = dist(nearest) - updatingCentroid(1, numDim+2) - ...
                                transpose(centroids(nearest, numDim+2));
                        if all(afterDist > 0)
                            % It is possible to grow independently of
                            % neighbors classes
                            centroids(id, :) = updatingCentroid;
                        else
                            % The radius touches or trespass at least
                            % another radius after growing. It is necessary
                            % to find those neighbors whose radius are
                            % touched
                            touchedNeighbors = afterDist <= 0;                                                        
                            if all(centroids(nearest(touchedNeighbors), numDim+1) == centroids(id, numDim+1))
                                % If all the touched neighbors are from the
                                % same class
                                newCtr = zeros(1, numDim+3);
                                newCtr(1, numDim+1) = centroids(id, numDim+1);
                                newCtr(1, numDim+2) = max(centroids([id nearest(touchedNeighbors)], numDim+2));
                                newCtr(1, end) = 1;
                                for d=1:numDim
                                    newCtr(1, d) = mean(centroids([id nearest(touchedNeighbors)], d));
                                end
                                % Inserting new centroid and erasing the
                                % other two from the set
                                centroids = [centroids; newCtr];
                                centroids([id nearest(touchedNeighbors)], :) = [];
                                % Updating current class indices for the
                                % while loop
                                currentClsIdx = find(centroids(:, numDim+1) == currentCls);
                            else
                                % At least one of the touched centroids is
                                % from a different class. For
                                % simplification, in this case we just
                                % change their growth flag
                                centroids([id nearest(touchedNeighbors)], end) = 0;
                            end
                        end
                    end
                end
                % Updating i flag value
                i = i+1;
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