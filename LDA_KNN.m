clear
close all
clc

%% setup
load('face.mat');
rng(1)

% dimensions
width = 46;
height = 56;

% set some Ns
N = size(X, 2);
N_faces_per_person = 10;
N_people = N / N_faces_per_person;
N_features = size(X, 1);

% generate train/test split
train = 7;
test = N_faces_per_person - train;

% create logical (boolean) indices to be used for splitting l, X
train_split = [ones(1,train), zeros(1,test)];

% shuffle train/test images
train_split = train_split(randperm(N_faces_per_person));

train_indices = logical(repmat(train_split, [1, N_people]));

% split dataset, train is the ones in the indices, test is the inverse
l_train = l(:, train_indices);
l_test = l(:, ~train_indices);

X_train = X(:, train_indices);
X_test = X(:, ~train_indices);

mean_train_image = mean(X_train, 2);
mean_class_images = zeros(N_features, N_people);

%% calculate S_W, S_B
S_W = zeros(N_people, N_features, N_features);

% create a mean image for each person
for i = 1:N_people
    index = (i - 1) * train;
    current_train = X_train(:, index + 1:(index + train));
    
    mean_class_image = mean(current_train, 2);
    mean_class_images(:, i) = mean_class_image;
    
    diffed_class_image = current_train - mean_class_image;
    
    % form within-class scatter matrix
    S_W(i, :,:) = diffed_class_image * diffed_class_image';
end

S_W = reshape(sum(S_W, 1), [N_features, N_features]);
diffed_class_mean_images = mean_class_images - mean_train_image;

S_B = (diffed_class_mean_images)*(diffed_class_mean_images)';

S_T = S_B + S_W;

M_pca = 312;
% M_pca = rank(S_W);

[W_pca, D_pca] = eigs(S_T, M_pca);

intermediate = (W_pca' * S_W * W_pca)\(W_pca' * S_B * W_pca);

accuracy_mldas = zeros(1, 51);

for M_lda = 1:51
    % M_lda = rank(S_B);

    [W_lda, D_lda] = eigs(intermediate, M_lda);

    W_opt = W_pca * W_lda;

    %% testing W_opt
    proj_train = (X_train - mean_train_image)' * W_opt;
    proj_test = (X_test - mean_train_image)' * W_opt;

    Idx = knnsearch(proj_train, proj_test);

    % use l to determine if we got the right person, results(n) = 0 if we did
    predicted_class = l_train(Idx);    
    results = abs(l_test - predicted_class);

    % every time we get it wrong, make it a positive number, then turn it
    % into a logical array, invert and sum it to get the number of correct
    % test examples, divide by the total number of tests and turn into a
    % percentage
    accuracy = sum(~logical(results))/length(l_test)*100;
    accuracy_mldas(M_lda) = accuracy;
end

accuracy_pure_pca = 63.461538461538460;

plot(accuracy_mldas)
hold on
plot(ones(1, 51)* accuracy_pure_pca)
xlabel('$M_{LDA}$', 'Interpreter', 'latex')
ylabel('Accuracy (%)')
legend({'Fisherfaces with varying $M_{LDA}$', 'Pure PCA'}, 'Interpreter', 'latex', 'Location', 'best')

%%
% %% Plot Confusion Chart
% 
% figure
% confusionchart(l_test, predicted_class);
% % print('./figures/q3_confusionmatrix_PCA_LDA.png', '-dpng')
% 
% %% Plot Error case
% 
% img_width = 46;
% img_height = 56;
% 
% figure
% subplot(1, 2, 1);
% img_1 = mat2gray(reshape(X_test(:,3), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
% imshow(img_1, 'InitialMagnification', 'fit');
% title('Test image 3 from person 1');
% 
% subplot(1, 2, 2);
% img_1 = mat2gray(reshape(X_train(:,111), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
% imshow(img_1, 'InitialMagnification', 'fit');
% title('Training image 111 from person 16');
% 
% colormap(gray(255));
% % print('./figures/q3_error_case_PCA_LDA.eps', '-depsc','-tiff')

%% Plot Success case

img_width = 46;
img_height = 56;

figure
subplot(1, 2, 1);
img_1 = mat2gray(reshape(X_test(:,143), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
imshow(img_1, 'InitialMagnification', 'fit');
title('Test image 143 from person 48');

subplot(1, 2, 2);
img_1 = mat2gray(reshape(X_train(:,336), [img_height, img_width])); % takes the average face vector (D x 1) and changes it to an image matrix (W x H)
imshow(img_1, 'InitialMagnification', 'fit');
title('Training image 336 from person 48');

colormap(gray(255));
print('./figures/q3_success_case_PCA_LDA.png', '-dpng')

%% Plot 15 Biggest Fisherfaces
figure
h = zeros(1, 15);
for i = 1:15
    h(i) = subplot(3, 5, i);
    img_1 = mat2gray(reshape(W_opt(:,i), [img_height, img_width])); 
    imshow(img_1, 'InitialMagnification', 'fit');
    title( ['Fisherface ' num2str(i)]);
end
colormap(gray(255));
print('./figures/q3_fisherfaces.png', '-dpng')
