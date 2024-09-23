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
title('Gaussian Process Regression');
xlabel('X');
ylabel('Y');

set(gcf,'color','w'); % gcf stands for 'get current figure'

scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
filename = fullfile(scriptDir, 'simple_gp_demo.gif');

for i = 1:n
    % Update the kernel matrix with new training point
    for j = 1:i
        K(i, j) = RBF_kernel(X(i), X(j));
        K(j, i) = K(i, j); % symmetric matrix
    end
    
    % Add noise variance to the diagonal elements
    K(1:i, 1:i) = K(1:i, 1:i) + 0.1^2 * eye(i);
    
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
    
    % Plot the results
    cla; % clear current axes
    hold on;
    fill([X_test; flipud(X_test)], ...
         [mu_test - 2 * std_test; flipud(mu_test + 2 * std_test)], ...
         [0.9, 0.9, 0.9], 'EdgeColor', 'none','FaceAlpha',0.5); % 95% CI
    plot(X_test, mu_test, 'r', 'LineWidth', 2); % Predictive mean
    scatter(X(1:i), Y(1:i), 50, 'filled','MarkerFaceColor','blue'); % Plot the training points
    hold off;
    legend('95% CI', 'GPR mean', 'Training points', 'Location', 'Best');

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
