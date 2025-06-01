function RESULT = NFoldTest2(X, Y, N)
rng(75);
% N-fold Cross-Validation
% Input:
% X - Feature matrix (number of samples × number of features)
% Y - Label matrix (number of samples × 1 column, each row has two cases: 0 or 1 representing two categories)
% N - Number of folds (e.g., 5-fold, 10-fold)

% Convert labels to 1D labels: 0 -> 1, 1 -> 2
Y_1d = Y + 1;

% Keep data in original order (no shuffling)
features = X;
labels = Y_1d;

% Extract indices of two classes (stratified sampling)
idx1 = find(labels == 1);
idx2 = find(labels == 2);

% Cross-validation loop
num_samples = length(labels);
fold_size = floor(num_samples / N); % Samples per fold
cv_accuracy = zeros(N, 1);        % Store test accuracy for each fold
cv_precision = zeros(N, 2);       % Store precision for each class in each fold
cv_sensitivity = zeros(N, 2);     % Store sensitivity for each class in each fold
cv_f1 = zeros(N, 2);             % Store F1 score for each class in each fold
% figure('Position', [100, 100, 800, 800]); % Set window size
for fold = 1:N
    fprintf('\n===== Fold %d/%d =====\n', fold, N);

    % Split training and test sets for current fold (stratified sampling to maintain class ratios)
    test_start1 = (fold-1)*floor(length(idx1)/N) + 1;
    test_end1 = fold*floor(length(idx1)/N);
    if fold == N
        test_end1 = length(idx1);
    end
    test_idx1 = idx1(test_start1:test_end1);

    test_start2 = (fold-1)*floor(length(idx2)/N) + 1;
    test_end2 = fold*floor(length(idx2)/N);
    if fold == N
        test_end2 = length(idx2);
    end
    test_idx2 = idx2(test_start2:test_end2);

    test_idx = [test_idx1; test_idx2]; % Combine test set indices
    train_idx = setdiff(1:length(Y), test_idx); % Training set indices

    % Extract features and labels for current fold
    train_features = features(train_idx, :);
    train_labels = labels(train_idx, :);
    test_features = features(test_idx, :);
    test_labels = labels(test_idx, :);

    % Model parameter initialization
    tau_m = 10;    % Membrane time constant (ms)
    V_thresh = 5;  % Firing threshold
    V_rest = 0;    % Resting potential
    w_inh = -0.1;  % Inhibitory strength
    T = 100;       % Time window (ms)
    dt = 1;        % Time step (ms)
    num_steps = T/dt;
    % Initialize weights for two classes
    w = randn(size(train_features, 2), 2);
    % Train the model
    num_epochs = 100;
    train_accuracy = zeros(num_epochs, 1);
    for epoch = 1:num_epochs
        correct_count = 0;
        for i = 1:size(train_features, 1)
            x = train_features(i, :)';
            XLENGTH = floor(length(x) / num_steps);
            % Simulate dynamics of two neurons
            V = repmat(V_rest, 2, 1);
            n = zeros(2, 1); % Spike count
            for t = 1:num_steps
                wtemp = w((t-1)*XLENGTH+1:t*XLENGTH,:);
                xtemp = x((t-1)*XLENGTH+1:t*XLENGTH,:);
                f = max(0, wtemp' * xtemp); % Input current
                for j = 1:2
                    inh_term = 0;
                    for k = 1:2
                        if k ~= j
                            inh_term = inh_term + w_inh * n(k); % Inhibitory input from other neurons
                        end
                    end
                    % Membrane potential dynamics
                    V(j) = V(j)*exp(-dt/tau_m) + (1-exp(-dt/tau_m))*f(j) + inh_term;
                    if V(j) >= V_thresh % Spike occurrence
                        n(j) = n(j) + 1;
                        V(j) = V_rest; % Reset potential after spike
                    end
                end
            end

            % Decision: predict class with highest spike count
            [~, prediction] = max(n);

            if prediction == train_labels(i)
                correct_count = correct_count + 1;
            else
                % Update weights using Hebb's rule
                learning_rate = 1e-4;
                for j = 1:2
                    if j == train_labels(i)
                        w(:, j) = w(:, j) + learning_rate * x; % Strengthen connection for correct class
                    else
                        w(:, j) = w(:, j) - learning_rate * x; % Weaken connection for incorrect classes
                    end
                end
            end
        end

        train_accuracy(epoch) = correct_count / size(train_features, 1);
    end
    mean_train_accuracy(fold) = mean(train_accuracy);
    fprintf('Fold %d Train Acc: %.2f%%\n', fold, mean_train_accuracy(fold)*100);

    % Test set evaluation
    predictions = zeros(size(test_features, 1), 1);
    for i = 1:size(test_features, 1)
        x = test_features(i, :)';
        V = repmat(V_rest, 2, 1);
        n = zeros(2, 1);
        for t = 1:num_steps
            wtemp = w((t-1)*XLENGTH+1:t*XLENGTH,:);
            xtemp = x((t-1)*XLENGTH+1:t*XLENGTH,:);
            f = max(0, wtemp' * xtemp);
            for j = 1:2
                inh_term = 0;
                for k = 1:2
                    if k ~= j
                        inh_term = inh_term + w_inh * n(k);
                    end
                end
                V(j) = V(j)*exp(-dt/tau_m) + (1-exp(-dt/tau_m))*f(j) + inh_term;
                if V(j) >= V_thresh
                    n(j) = n(j) + 1;
                    V(j) = V_rest;
                end
            end
        end
        [~, predictions(i)] = max(n);
    end

    % Calculate evaluation metrics using confusion matrix
    confusionMat = zeros(2, 2);
    for true_class = 1:2
        for pred_class = 1:2
            confusionMat(true_class, pred_class) = sum(test_labels == true_class & predictions == pred_class);
        end
    end

    numTestSamples = length(test_labels);
    for class = 1:2
        TP = confusionMat(class, class); % True Positives
        FP = sum(confusionMat(:, class)) - TP; % False Positives
        FN = sum(confusionMat(class, :)) - TP; % False Negatives
        TN = numTestSamples - TP - FP - FN; % True Negatives

        cv_accuracy(fold) = (sum(diag(confusionMat))) / numTestSamples; % Overall accuracy
        cv_precision(fold, class) = TP / (TP + FP + eps); % Precision
        cv_sensitivity(fold, class) = TP / (TP + FN + eps); % Sensitivity (Recall)
        cv_f1(fold, class) = 2*cv_precision(fold, class)*cv_sensitivity(fold, class) / ...
            (cv_precision(fold, class) + cv_sensitivity(fold, class) + eps); % F1 Score
    end
    % subplot(5, 2, fold);
    % confusionchart(confusionMat,{'L','H'});
    % title(sprintf('Fold %d (Acc=%.2f%%)', fold, cv_accuracy(fold)*100));
    fprintf('Fold %d Test Acc: %.2f%%\n', fold, cv_accuracy(fold)*100);

    cm{fold} = confusionMat;
end

% ======================================================================================
% Output cross-validation results

fprintf('\n===== Cross-validation Results (%d-fold) =====\n', N);
fprintf('Average Training Accuracy: %.2f(±%.2f)%%\n', mean(mean_train_accuracy)*100, std(mean_train_accuracy)*100);
fprintf('Best Training Accuracy: %.2f%%\n', max(mean_train_accuracy)*100);
fprintf('Worst Training Accuracy: %.2f%%\n', min(mean_train_accuracy)*100);
fprintf('Average Test Accuracy: %.2f(±%.2f)%%\n', mean(cv_accuracy)*100, std(cv_accuracy)*100);
fprintf('Best Test Accuracy: %.2f%%\n', max(cv_accuracy)*100);
fprintf('Worst Test Accuracy: %.2f%%\n', min(cv_accuracy)*100);
for class = 1:2
    fprintf('Class %d Average Test Precision: %.2f%%\n', class, mean(cv_precision(:, class))*100);
    fprintf('Class %d Average Test Sensitivity: %.2f%%\n', class, mean(cv_sensitivity(:, class))*100);
    fprintf('Class %d Average Test F1 Score: %.2f%%\n', class, mean(cv_f1(:, class))*100);
end

% Return results
RESULT = {cv_f1*100};
end