% Main VDBC Script to initialize VDBC execution

%% Loading data, variables and objects
%=============================================
%=============DATA============================    
numFolds = 5;

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
    
    [dataF, dataTNum, numCls, numDim] = getDB(db);
    
    disp(db);
    results = VDBC(dataF, dataTNum, numCls, numDim, numFolds);
    
    fileName = strcat(db, '_', num2str(numFolds), 'folds');
    
    save(fileName, 'results')
end
