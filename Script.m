data = load("cifar-10-data.mat");
doubleData = double(data.data);
%Convert the image data set into double
NormalisedImageData = doubleData / 255;
%Normalise the converted image data
RandomisationSeed = 37999524;
%Store the randomisation see in a variable
rng(RandomisationSeed,"twister"); %twister = Mersenne Twister(MT)
% function which chooses a seed number and the type of algorithm MT
randomNumbers = randperm(60000, 4);
%Generated 4 random numbers to create the subplot grid, which will hold .
 for i = 1 : 4
     subplot(1,4,i);
     reportImage = squeeze(NormalisedImageData(randomNumbers(i),:,:,:));
     imagesc(reportImage);
     title(label_names(labels(randomNumbers(i))+1));
     % title
 end
 rng(RandomisationSeed, "twister");
 classes = randperm(9,3);
% Results in 7, 9, and 4
% 7 = horse, 9 = truck, 4 = deer
% Now extract all images that are either, horse, truck or deer.

classData = zeros(18000,32, 32, 3);
% Creates a matrix with the dimensions 18k x 32 x 32 x 3
classLabels = zeros(18000,1);
% Creates a vector with the dimensions 18k x 1
count = 0;
%Counter for the for loop

%For loop which traverses through the image data set, and checks if the
%members of the labels data set fall within one of the 3 classes. If they
%do fall within one of the 3 classes then the image is copied into the
%class data and the label number is copied into the classLabels matrix
for i = 1:size(NormalisedImageData, 1)
    % the size(NormalisedImageData, 1) gives us the size of the Image
    % dataset (60k images)
    if ismember(labels(i), classes)
    count = count + 1;
        classData(count,:,:,:) = NormalisedImageData(i,:,:,:);
        classLabels(count,:) = labels(i,:);
    end
end 

rng(RandomisationSeed, "twister");

dataSetValues = 1:18000;
%Creates a dataset which ranges from 1 to 18000, used to satisfy the
%setdiff function,

trainig_index = randperm(18000, 9000)';
%stores  the data in the training_Index for training purposes, splitting
%the total number of images in half

testing_index = setdiff(dataSetValues, trainig_index)';
% Uses the variable datasetValues as a range, to subtract all numbers
% within the training_Index thus resulting in the other half of the image
% indexes

% Creation of training & testing datasets and labels
training_data = classData(trainig_index,:,:,:);
testing_data = classData(testing_index,:,:,:);
training_labels = classLabels(trainig_index);
testing_labels = classLabels(testing_index);

% Reshape the testing & trainingdata into 9000x3071 matrices
reshapedTestingData = reshape(testing_data, 9000, 3072);
reshapedTrainingData = reshape(training_data, 9000, 3072);

% 4.1 Training 2 of my Models
% CUSTOM KNN code

% EUCLIDEAN DISTANCE FOR LOOP
k = 5;
tic
categorical_Labels = categorical(testing_labels);
    %Empty array which will hold predictions
Euclidean_testing_Predictions = categorical.empty(size(reshapedTestingData, 1), 0);
    %Iterate over the testing sampe
    for i = 1:size(reshapedTestingData, 1);
        % Retrieve the current vector being tested
        comp1 = reshapedTestingData(i,:);
        % stores the training data
        comp2 = reshapedTrainingData;
        %create an array to store the distance, size of the trainingdata
        distanceArray = zeros(size(reshapedTrainingData, 1), 1);
        distanceArray = sqrt(sum((comp2 - comp1).^2,2));
        [~,ind] = sort(distanceArray);
        ind = ind(1:k);
        labs = categorical(training_labels(ind));
        Euclidean_testing_Predictions(i,1) = categorical(mode(labs));
        disp(i)
    end
    
    %Accuracy Calculations
    Eucliedean_Time = toc;
    Euclidean_Correct_predictions = sum(categorical_Labels == Euclidean_testing_Predictions);
    Euclidean_Accuracy = Euclidean_Correct_predictions/size(testing_labels,1);
   

Cosine_testing_Predictions = categorical.empty(size(reshapedTestingData, 1), 0);
% COSINE DISTANCE FOR LOOP
tic
    % Empty array which will hold predictions
    % Iterate over the testing sampe
    for i = 1:size(reshapedTestingData, 1);
        % Retrieve the current vector being tested
        comp1 = reshapedTestingData(i,:);
        % stores the training data
        comp2 = reshapedTrainingData;
        % create an array to store the distance, size of the trainingdata
        distanceArray = zeros(size(reshapedTrainingData, 1), 1);
        distanceArray = 1 - sum(comp2.* comp1, 2) ./ (sqrt(sum(comp2.^2, 2)) .* sqrt(sum(comp1.^2, 2)));
        [~,ind] = sort(distanceArray);
        ind = ind(1:k);
        labs = categorical(training_labels(ind));
        Cosine_testing_Predictions(i,1) = categorical(mode(labs));
        disp(i)
    end
Cosine_Time = toc;
    Cosine_Correct_predictions = sum(categorical_Labels == Cosine_testing_Predictions);
    Cosine_Accuracy = Cosine_Correct_predictions/size(testing_labels,1);
    % disp(['Cosine Data:     Accuracy: ', num2str(Cosine_Accuracy), ' Time: ',  num2str(Cosine_Time)]);

%DTREE ALGO

tic
DTreeModel = fitctree(reshapedTrainingData, training_labels);
DTREEPred = predict(DTreeModel,reshapedTestingData);
DTREE_Time = toc;
DTREE_Correct_predictions = sum(testing_labels == DTREEPred);
DTREE_Accuracy = DTREE_Correct_predictions/size(testing_labels, 1);
    

% SVM 
tic
SVM_Model = fitcecoc(reshapedTrainingData, training_labels);
SVM_Pred = predict(SVM_Model, reshapedTestingData);
SVM_Time = toc;
    SVM_Correct_predictions = sum(testing_labels == SVM_Pred);
    SVM_Accuracy = SVM_Correct_predictions/size(testing_labels,1);

    subplot(1,4,1);
    Euclidean_Confusion_Chart = confusionchart(categorical_Labels, Euclidean_testing_Predictions);
    % Euclidean_Confusion_Chart.Title('Euclidean');
    subplot(1,4,2);
    Cosine_Confusion_Chart = confusionchart(categorical_Labels, Cosine_testing_Predictions);
    % Cosine_Confusion_Chart.Title('Cosine');
    subplot(1,4,3);
    DTREE_Confusion_Chart = confusionchart(testing_labels, DTREEPred);
    % DTREE_Confusion_Chart.Title('DTREE');
    subplot(1,4,4);
    SVM_Confusion_Chart = confusionchart(testing_labels, SVM_Pred);
    % SVM_Confusion_Chart.Title('SVM');

    disp(['Euclidean Data:     Accuracy: ', num2str(Euclidean_Accuracy), ' Time: ',  num2str(Eucliedean_Time)]);
    disp(['Cosine Data:     Accuracy: ', num2str(Cosine_Accuracy), ' Time: ',  num2str(Cosine_Time)]);
    disp(['DTREE Data:     Accuracy: ', num2str(DTREE_Accuracy), ' Time: ',  num2str(DTREE_Time)]);
    disp(['SVM Data:     Accuracy: ', num2str(SVM_Accuracy), ' Time: ',  num2str(SVM_Time)]);
save('cw1.mat', 'Euclidean_Accuracy', "Eucliedean_Time", "Euclidean_Confusion_Chart", "Cosine_Accuracy", "Cosine_Time", "Cosine_Confusion_Chart", ...
    "DTREE_Accuracy", "DTREE_Time", "DTREE_Confusion_Chart", "SVM_Accuracy", "SVM_Time", "SVM_Confusion_Chart");