% Main VDBC Script to initialize VDBC execution

%% Loading data, variables and objects
%=============================================
%=============DATA============================    
numFolds = 5;

% Vector to store MAUC values
results(numFolds).mauc = 0;
results(numFolds).numProt = 0;

for dataSet = 1:21
    switch dataSet
    % Possible data sets
        case 1
            db = 'abalone';             % 23 classes
            
        case 2
            db = 'arrhythmia';          % 13 classes
            
        case 3
            db = 'balanceScale';        % 03 classes
            
        case 4
            db = 'carEval';             % 04 classes
            
        case 5
            db = 'contraceptive';       % 03 classes
            
        case 6
            db = 'dermatology';         % 06 classes
            
        case 7
            db = 'ecoli';               % 08 classes
            
        case 8
            db = 'gene';                % 03 classes
            
        case 9
            db = 'glass';               % 06 classes
            
        case 10
            db = 'hayes';               % 03 classes
            
        case 11
            db = 'horse';               % 03 classes
            
        case 12
            db = 'nursery';             % 05 classes
            
        case 13
            db = 'page-blocks';         % 05 classes
            
        case 14
            db = 'post-operative';      % 03 classes
            
        case 15
            db = 'satimage';            % 06 classes
            
        case 16
            db = 'shuttle';             % 07 classes
            
        case 17
            db = 'soybean';             % 15 classes
            
        case 18
            db = 'thyroid';             % 03 classes
            
        case 19
            db = 'wine';                % 03 classes
            
        case 20
            db = 'yeast';               % 10 classes
            
        case 21
            db = 'zoo';                 % 07 classes
            
    end
    
    % Pre-process==========================================================
    [dataF, dataTNum, numCls, numDim] = getDB(db);
    
    % Separating data into k folds
    foldIdx = kfoldIndices(numCls, dataTNum, numFolds);
    
    disp(db);
    
    for k=1:numFolds
        disp(strcat('ITERAÇÃO', num2str(k)));
        
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
        
        [results(k).mauc, results(k).numProt] = ...
            VDBC(trainSet, trainTargets, testSet, testTargets, numDim, numCls);
    end
    
    fileName = strcat(db, '_', num2str(numFolds), 'folds');
    
    save(fileName, 'results')
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