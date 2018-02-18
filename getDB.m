function [dataF, dataTNum, numCls, numDim] = getDB(name)
        
    datafile = dlmread(strcat(name, '.dt'));
    
    % Find characteristics of data
    dline = datafile(1,:);
    % Find the number of features and classes
    numDim = dline(2); numCls = dline(3);
            
    % Creating feature and target matrices
    dataF = datafile(2:end, 1:numDim);
    dataTNum = datafile(2:end, end);
         
    % Some classes have less than 2 members. Those will be ignored
    classCount = histc(dataTNum, unique(dataTNum));
    smallCls = find(classCount < 2);
            
            
    if size(smallCls, 1) >= 1
        for i=1:size(smallCls, 1)
            toEraseIdx = find(dataTNum == smallCls(i));
            dataTNum(toEraseIdx) = [];
            dataF(toEraseIdx, :) = [];
        end
            
        % Updating classes numbers
        remainCls = unique(dataTNum); numCls = size(remainCls, 1);
        newClsNum = 1:numCls;
        for i=1:numCls
            dataTNum(dataTNum == remainCls(i)) = newClsNum(i);
        end
    end
    
end
