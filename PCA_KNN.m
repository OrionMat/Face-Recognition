clear
close all
clc

load('face.mat');
rng(1) 

%% Parameters

[D_features, N_faces] = size(X);                            % N_faces is number of faces (520). D_features is number of features/dimensions (2576)
N_faces_per_person = 10;                                    % sets the number of faces per person
N_people = N_faces / N_faces_per_person;
train_per_person = 7;                                       % sets the split between number of training faces and number of test faces (per person)
test_per_person = N_faces_per_person - train_per_person;
M_max = train_per_person * N_people;                        % M is number of eigen values used (M_max is largest possible M = number of training faces)
img_width = 46;
img_height = 56;

%% Splitting data set (for cross-validation)

train_split = [ones(1, train_per_person),   zeros(1, test_per_person)];
train_split = train_split(randperm(N_faces_per_person));    % shuffles the  train and test images
train_indices = logical(repmat(train_split, 1, N_people));  % logical array that chooses which coloumns should be in the training data

l_train = l(:, train_indices); 
l_test = l(:, ~train_indices); 
X_train = X(:, train_indices);
X_test = X(:, ~train_indices);


%% PCA (S = 1/N * A * A')

X_train_avg = mean(X_train, 2); % compute average face vector (the mean value of each row/feature)  
A = X_train - X_train_avg;      % subtract average face vector from each face/coloumn 
A_test = X_test - X_train_avg;

tic
S = (1/N_faces)*A*(A');         % Computing covarience matrix
[U, Eval] = eigs(S, N_faces);   % Computing M eigen-vectors and eigen-values of covarience matrix
toc

Eval = real(sum(Eval, 1));

%% Plot Eigenvalues

figure
Eval_plot = stem(Eval);         % shows the value of each eigen value. largest eigen values correspond to best eigen vectors as data has most varience in these directions
set(Eval_plot, 'Marker', 'none');
title('Value of each eigen value in descending order of magnitude')
xlabel('Eigenvalue index') 
ylabel('Value') 
print -deps Eval_plot

%% Choosing M 

M = find(0.99*sum(Eval) < cumsum(Eval));    % finds all indicies where the cumalutive sum of the eigen values are greater than 99% of the total sum of the eigen values
M = M(1);                                   % M is the first element that meets this criteria 

%% PCA_low (S = 1/N * A' * A)

tic
S_low = (1/N_faces)*(A')*A;     % low dimensional computation of the eigenspace
[V_low, Eval_low] = eigs(S_low, M_max);
U_low = A*V_low;
U_low = normc(U_low);
toc

Eval_low = real(sum(Eval_low, 1));

%% Reconstruction while varying number of eigen faces

%W_project = U'*A;          % high dimension reconstruction (less error)
%X_train_rec = X_train_avg + U*W_project;

X_face_rec_mat = zeros(D_features, 18);
rec_accuracy_mat = zeros(18, M_max);
Eval_index = zeros(1, 18);
j = 1;
for i = 20:20:M_max
    W_project_low = (U_low(:, 1:i))'*A;
    X_train_rec_low = X_train_avg + U_low(:, 1:i)*W_project_low;
    
    X_face_rec_mat(:,j)=X_train_rec_low(:,1);
    
    X_train_rec_error = abs(X_train - X_train_rec_low);
    rec_accuracy = 100 - ((sum(X_train_rec_error) ./ sum(X_train)) * 100);
    rec_accuracy_mat(j,:) = rec_accuracy;
    Eval_index(j) = i;
    j = j +1;
end

rec_accuracy_avg = mean(rec_accuracy_mat, 2);

%% plot fce renonstructions

figure
h = zeros(1, 6);
for i = 1:6
    h(i) = subplot(2, 3, i);
    img_1 = reshape(X_face_rec_mat(:,3*i), [img_height, img_width]); 
    image(img_1, 'Parent', h(i));   
    title( [num2str(i*60) ' Eigen faces']);
end
colormap(gray(255));
print -deps face_recs

%% plot reconstruction accuracy

figure
stem(Eval_index, rec_accuracy_avg);  
title('Reconstruction accuracy as against the number of Eigenfaces used')
xlabel('Number of Eigenfaces used in reconstruction') 
ylabel('Reconstruction accuracy (%)') 
print -deps Rec_acc


%% KNN classification using PCA

predicted_class_matrix = zeros(M_max, test_per_person * N_people);  % stores predicted classes for each face
accuracy_matrix = zeros(1, M_max);                                  % stores accuracies for each number of eigen values used

for i = 1:M_max
 
    X_train_proj = (A')*U_low(:,1:i);         % Projecting normalised faces onto the face-space. Rows are the projections of normalised faces onto the eigenface (row 1 is projection of face 1)
    X_test_proj = (A_test')*U_low(:,1:i);
    
    Idx = knnsearch(X_train_proj, X_test_proj);     % Perform a knnsearch between X_train_proj and X_test_proj to find indices of nearest neighbor and puts in coloumn vector
    
    predicted_class = l_train(Idx);
    prediction_results = predicted_class == l_test;
    prediction_accuracy = (sum(prediction_results)*100)/(test_per_person * N_people);
    
    predicted_class_matrix(i, :) = predicted_class;
    accuracy_matrix(i) = prediction_accuracy;
end

%X_train_rec = X_train_avg + U_low*X_train_proj';
%Error_rec = vecnorm(X_train - X_train_rec);

%% Plot average training face

figure
img_1 = reshape(X_train_avg, [img_height, img_width]); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
image(img_1);
colormap(gray(255));
print -deps PCA_avg_face

%%

figure
plot(accuracy_matrix);  % shows can stop at around 100 eigen values
title('Classification accuracy against number of eigen values used')
xlabel('Eigen values used (M)') 
ylabel('Accuracy (%)') 

figure
%confusionchart(l_test, predicted_class_matrix(100, :)); % selects the results obtained using 100 eigen vectors and creates a confusion chart from them

%% Classification using reconstruction

Error_rec = zeros(N_people, test_per_person * N_people);

for i = 1:N_people
    index = (i - 1) * train_per_person;
    
    class_train = X_train(:, index + 1:(index + train_per_person));    % takes each class of person
    
    class_avg = mean(class_train, 2);
    class_normalized = class_train - class_avg;
    S_class = class_normalized'*class_normalized;
    
    [V_class, Eval_class] = eigs(S_class, train_per_person -1);
    U_class = class_normalized*V_class;
    U_class = normc(U_class);
    
    test_class_norm = X_test - class_avg;
    test_class_projection = U_class'*test_class_norm;
    
    X_test_rec = class_avg + U_class*test_class_projection;    
    test_difference = X_test - X_test_rec;
    
    Error_rec(i, :) = vecnorm(test_difference);
end

[Mins, I] = min(Error_rec, [], 1);

figure
%confusionchart(l_test, I);

results_reconst = l_test == I;

accuracy_reconst = sum(results_reconst)/length(l_test)*100;


%% Plot Error case

img_width = 46;
img_height = 56;

figure
subplot(1, 2, 1);
img_1 = mat2gray(reshape(X_test(:,31), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
imshow(img_1, 'InitialMagnification', 'fit');
title('Test image 31 from person 11');

subplot(1, 2, 2);
img_1 = mat2gray(reshape(X_train(:,123), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
imshow(img_1, 'InitialMagnification', 'fit');
title('Training image 123 from person 18');

colormap(gray(255));
print -depsc PCA_NN_Error


