function [mauc] = VDBC(trainSet, trainTargets, testSet, testTargets, numDim, numCls)
% VDBC algorithm with modification. Let C = {c1, c2, ..., cn} be the set of
% classes and p(c_n) be the probability of a sample belong to class n. The 
% average of all classes' probabilities is calculated and then compared to 
% each single probability. The class with single probability lesser than 
% the average is chosen to be increased via SMOTE.
    
    % Finding classes to be increased through SMOTE
    clsProbabilities = zeros(1, numCls);
    for i=1:numCls
        clsProbabilities(i) = size(find(trainTargets == i), 1)/size(trainSet, 1);
    end
    toIncrease = find(clsProbabilities < mean(clsProbabilities));
    [trainSet, trainTargets] = oversample(trainSet, trainTargets, toIncrease, numDim);
    
    centroids = [];
    
    centroids = train(trainSet, trainTargets, numDim, centroids);
    output = testing(testSet, centroids);
    % size(unique(testTargets), 1) ---> sometimes not all classes are
    % present on test set, so it is passed the size of present classes
    mauc = calculateMAUC(output, testTargets, size(unique(testTargets), 1));
end

function [trainSet, trainTargets] = oversample(trainSet, trainTargets, toIncrease, numDim)
% Function to oversample selected classes

    addSet = []; addTargets = [];
    for i=1:size(toIncrease, 2)
        class = toIncrease(i);
        instances = trainSet(trainTargets == class, :);
        
        if size(instances, 1) > 1
            % One new synthetic instance between each pair of instances
            for n=1:size(instances,1)-1
                for m=n+1:size(instances,1)
                    newInst = zeros(1, numDim);
                    for d=1:numDim
                        newInst(1, d) = mean([instances(n, d) instances(m, d)]);
                    end
                    addSet = [addSet; newInst];
                    addTargets = [addTargets; i];
                end
            end
        else
            % There is only one training instance. In this case there will 
            % be 2 * numDim neighbors two for each dimension, + and - half 
            % of its mean value
            for d=1:numDim
                meanDim = mean(trainSet(:, d));
                        
                newInsts = zeros(2, numDim);
                newInsts(1,:) = instances; 
                newInsts(2,:) = instances;
                        
                newInsts(1, d) = newInsts(1, d)+(0.5*meanDim);
                newInsts(2, d) = newInsts(2, d)-(0.5*meanDim);
                        
                addSet = [addSet; newInsts];
                addTargets = [addTargets; i; i];
            end
        end
    end
    
    trainSet = [trainSet; addSet]; trainTargets = [trainTargets; addTargets];
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