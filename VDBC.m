function [mauc] = VDBC(trainSet, trainTargets, testSet, testTargets, numDim, numCls)
% VDBC with the third modification. The Generalized Fisher Index (GFI) is
% calculated for each pair of classes. All pairs in which GFI value is less
% then 0.15 will receive synthetic instances between them.
    
    % Finding classes characteristics--------------------------------------
    cls(numCls).ind = [];
    cls(numCls).size = 0;
    cls(numCls).centroid = zeros(1, numDim);
    for c=1:numCls
        cls(c).ind = find(trainTargets == c);
        cls(c).size = size(find(trainTargets == c), 1);
        cls(c).centroid = mean(trainSet(cls(c).ind, :), 1);
    end
    %----------------------------------------------------------------------
    % Finding pairs of classes in which GFI is less than 0.15
    pairs = fisherIndPairs(trainSet, numCls, cls, 0.15);
    % Creating new instances between each pair of class
    newInstances = []; newTargets = [];
    for p=1:size(pairs, 1)
        cls1 = pairs(p, 1);
        cls2 = pairs(p, 2);
        % Values range for new instances
        minRange = min(cls(cls1).centroid, cls(cls2).centroid);
        maxRange = max(cls(cls1).centroid, cls(cls2).centroid);
        numPoints = max(cls(cls1).size, cls(cls2).size);
        
        [newInstances, newTargets] = ...
            createInstances(minRange, maxRange, numPoints, cls1, cls2, numDim);
    end
    
    trainSet = [trainSet; newInstances];
    trainTargets = [trainTargets; newTargets];
    % In case of duplicated instances, they will be erased
    [u, rows, ~] = unique(trainSet, 'rows');
    if size(u, 1) < size(trainSet, 1)
        indDupRows = setdiff(1:size(trainSet, 1), rows);
        trainSet(indDupRows, :) = []; trainTargets(indDupRows) = [];
    end
    
    % From now on the algorithm remains the same of original VDBC
    centroids = [];
    
    centroids = train(trainSet, trainTargets, numDim, centroids);
    output = testing(testSet, centroids);
    % size(unique(testTargets), 1) ---> sometimes not all classes are
    % present on test set, so it is passed the size of present classes
    mauc = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
end

function pairs = fisherIndPairs(trainSet, numCls, class, limit)
% Calculate GFI for each pair of classes and return them if the value is
% less than the specified limit
    
    pairs = [];

    for cls1 = 1:numCls-1
        for cls2 = cls1+1:numCls
            inds = [class(cls1).ind; class(cls2).ind];
            
            globalCentroid = mean(trainSet(inds, :), 1);
            
            numerator = ...
                (class(cls1).size * distance(transpose(class(cls1).centroid), transpose(globalCentroid))) + ...
                (class(cls2).size * distance(transpose(class(cls2).centroid), transpose(globalCentroid)));
            
            denominator = 0;
            for i=1:class(cls1).size
                denominator = denominator + distance(transpose(trainSet(class(cls1).ind(i), :)), transpose(class(cls1).centroid));
            end
            
            for i=1:class(cls2).size
                denominator = denominator + distance(transpose(trainSet(class(cls2).ind(i), :)), transpose(class(cls2).centroid));
            end
            
            if (numerator/denominator) < limit
                pairs = [pairs; cls1 cls2];
            end
            
        end
    end
end

function [set, targets] = createInstances(rangeMin, rangeMax, numPoints, cls1, cls2, numDim)
% Function to create instances between two class centroids    
    
    set = zeros(numPoints, numDim);
    targets = zeros(numPoints, 1);
    
    for d = 1 : numDim
        set(:, d) = rangeMin(d) + (rangeMax(d) - rangeMin(d))*rand(numPoints, 1);
    end
    
    for i=1:numPoints
        if rand() > 0.5
            targets(i) = cls1;
        else
            targets(i) = cls2;
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