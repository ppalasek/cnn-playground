% +---------------------------------------------------------------------+ %
% | Post-process result of cnn-playground framework                     | %

% |                                                                     | %
% +---------------------------------------------------------------------+ %
clear all; close all; clc;


%                                                                         %
%                                                                         %
% **Datasets**
%   - 'mnist'     : http://yann.lecun.com/exdb/mnist/
%   - 'cifar10'   : https://www.cs.toronto.edu/~kriz/cifar.html
%   - 'cifar100'  : https://www.cs.toronto.edu/~kriz/cifar.html
%   - 'svhn'      : http://ufldl.stanford.edu/housenumbers/
%
%
% **Architectures**
%
% *CCFFF*
%   - 'ccfff-ap'
%   - 'ccfff-ap-d'
%   - 'ccfff-mp'
%   - 'ccfff-mp-d'
% *CCFFSVM*
%   - 'ccffsvm-ap'
%   - 'ccffsvm-ap-d'
%   - 'ccffsvm-mp'
%   - 'ccffsvm-mp-d'
%

no_iters = 3;
results_dir = '../results';

dataset = 'cifar10';
num_epochs = 100;
batch_size = 256;

architectures = {'ccfff-ap', ...
                 'ccfff-ap-d', ...
                 'ccfff-mp', ...
                 'ccfff-mp-d', ...
                 'ccffsvm-ap', ...
                 'ccffsvm-ap-d', ...
                 'ccffsvm-mp', ...
                 'ccffsvm-mp-d'};

TRAIN_LOSS_MEAN = zeros(num_epochs, length(architectures));
VALID_LOSS_MEAN = zeros(num_epochs, length(architectures));
VALID_ACC_MEAN = zeros(num_epochs, length(architectures));
TEST_LOSS_MEAN = zeros(1, length(architectures));
TEST_ACC_MEAN =  zeros(1, length(architectures));
             
for j=1:length(architectures)    
    
    % Set architecture
    arch = architectures{j};
    
    TRAIN_LOSS = zeros(num_epochs, no_iters);
    VALID_LOSS = zeros(num_epochs, no_iters);
    VALID_ACC = zeros(num_epochs, no_iters);
    TEST_LOSS = zeros(1, no_iters);
    TEST_ACC = zeros(1, no_iters);
    
    for i=1:no_iters
        % ----------------------------
        % Validation loss and accuracy
        % ----------------------------
        % Filename format:
        % <dataset>_<architecture>_<num_epochs>_<batch_size>_valid_<iter>.results
        valid_results_filename = sprintf('%s/%s_%s_%d_%d_valid_%d.results', ...
                                         results_dir, dataset, arch, num_epochs, batch_size, i);
        valid_results = dlmread(valid_results_filename);
        train_loss = valid_results(:,1);  % 1st column: training loss
        valid_loss = valid_results(:,2);  % 2nd column: validation loss
        valid_acc = valid_results(:,3);   % 3rd column: validation accuracy (%)
        TRAIN_LOSS(:,i) = train_loss;
        VALID_LOSS(:,i) = valid_loss;
        VALID_ACC(:,i) = valid_acc;

        % ----------------------------------------
        % Final results (test loss, test accuracy)
        % ----------------------------------------
        % Filename format:
        % <dataset>_<architecture>_<num_epochs>_<batch_size>_test_<iter>.results
        test_results_filename = sprintf('%s/%s_%s_%d_%d_test_%d.results', ...
                                        results_dir, dataset, arch, num_epochs, batch_size, i);
        test_results = dlmread(test_results_filename);
        test_loss = test_results(1);  % 1st element: test loss
        test_acc = test_results(2);   % 2nd element: test accuracy (%)
        TEST_LOSS(i) = test_loss;
        TEST_ACC(i) = test_acc;
    end
    % Average results over epochs
    TRAIN_LOSS_MEAN(:,j) = mean(TRAIN_LOSS, 2);
    VALID_LOSS_MEAN(:,j) = mean(VALID_LOSS, 2);
    VALID_ACC_MEAN(:,j) = mean(VALID_ACC, 2);
    TEST_LOSS_MEAN(j) = mean(TEST_LOSS);
    TEST_ACC_MEAN(j) = mean(TEST_ACC);
    
end

%
%
%

epochs = 1:num_epochs;

% Figure: Train Loss
figure(1); hold on;
for j=1:length(architectures)
    arch = architectures{j};
    plot(epochs, TRAIN_LOSS_MEAN(:,j), '-');
end
title(sprintf('Dataset: %s | Train Loss', dataset));
xlabel('epoch');
ylabel('Train Loss');
legend(architectures{:});

% Figure: Validation Loss
figure(2); hold on;
for j=1:length(architectures)
    arch = architectures{j};
    plot(epochs, VALID_LOSS_MEAN(:,j), '-');
end
title(sprintf('Dataset: %s | Validation Loss', dataset));
xlabel('epoch');
ylabel('Validation Loss');
legend(architectures{:});

% Figure: Validation Accuracy
figure(3); hold on;
for j=1:length(architectures)
    arch = architectures{j};
    plot(epochs, VALID_ACC_MEAN(:,j), '-');
end
title(sprintf('Dataset: %s | Validation Accuracy', dataset));
xlabel('epoch');
ylabel('Validation Accuracy');
legend(architectures{:});



