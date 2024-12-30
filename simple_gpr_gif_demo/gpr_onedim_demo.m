% Generate sample data
X = -3 + 6 * rand(20, 1);
Y = sin(X) + 0.05 * randn(20, 1); % Observations Y include some noise
n = length(X);

%% Define the RBF kernel function
sigma = 1.0; % Bandwidth of the RBF kernel
RBF_kernel = @(x1, x2) exp(-norm(x1 - x2)^2 / (2 * sigma^2));
%%
K = zeros(n, n); % initialize the kernel matrix

% Test inputs for prediction
X_test = linspace(-3, 3, 100)';
m_test = length(X_test);

figure;
set(gcf,'color','w'); % gcf stands for 'get current figure'

scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
filename = fullfile(scriptDir, 'gpr_onedim_demo.gif');

for i = 1:n
    % Update the kernel matrix with new training point
    for j = 1:i
        K(i, j) = RBF_kernel(X(i), X(j));
        K(j, i) = K(i, j); % symmetric matrix
    end
    
    % Add noise variance to the diagonal elements
    K(1:i, 1:i) = K(1:i, 1:i) + 0.05^2 * eye(i);
    
    % Compute the kernel between test points and training points
    K_test = zeros(m_test, i);
    for k = 1:m_test
        for j = 1:i
            K_test(k, j) = RBF_kernel(X_test(k), X(j));
        end
    end
    
    % Compute predictive mean
    K_inv = inv(K(1:i, 1:i));
    mu_test = K_test * K_inv * Y(1:i);
    
    % Compute predictive variance
    K_test_test = zeros(m_test, m_test);
    for k = 1:m_test
        for j = 1:m_test
            K_test_test(k, j) = RBF_kernel(X_test(k), X_test(j));
        end
    end
    
    var_test = K_test_test - K_test * K_inv * K_test';
    std_test = sqrt(diag(var_test));
    
    % Identify test points that are not training points
    [isTrain, ~] = ismembertol(X_test, X(1:i), 1e-6);
    X_test_CI = X_test(~isTrain);
    mu_test_CI = mu_test(~isTrain);
    std_test_CI = std_test(~isTrain);
    %% Plot the results
    cla; % clear current axes
    hold on;
    fill([X_test_CI; flipud(X_test_CI)], ...
         [mu_test_CI - 2 * std_test_CI; flipud(mu_test_CI + 2 * std_test_CI)], ...
         [0.7, 0.7, 0.7], 'EdgeColor', 'none','FaceAlpha',0.5); % 95% CI
    plot(X_test, mu_test, 'r', 'LineWidth', 2); % Predictive mean
    scatter(X(1:i), Y(1:i), 50, 'filled','MarkerFaceColor','blue'); % Plot the training points
    hold off;
    legend('95% CI', 'GPR mean', 'Training points', 'Location', 'Best');
    title(['Gaussian Process Regression (N = ' num2str(i) ')']);
    xlabel('X');
    ylabel('Y');
    
    %% Generate gif
    drawnow;
    frame = getframe(gcf); % Capture figure frame
    im = frame2im(frame); % Convert frame to image
    [A,map] = rgb2ind(im,256); % Convert image to indexed image
    if i == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
    end

    pause(1);
end
