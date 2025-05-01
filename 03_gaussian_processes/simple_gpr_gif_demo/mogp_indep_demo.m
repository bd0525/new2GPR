% Copyright (c) 2025, Boyuan Deng
% All rights reserved.

%% Generate Sample Data
N_total = 20; % num of data points
x_full = -5 + 10 * rand(N_total, 1);
sigma_y = 0.1; % noise std for y
sigma_z = 0.1; % noise std for z

% Output z & y independent
y_full = sin(x_full) + sigma_y * randn(N_total, 1);
z_full = 0.5 * y_full + 0.2 * x_full + sigma_z * randn(N_total, 1);

% Test inputs for prediction
x_test = linspace(-5, 5, 100)';
N_test = length(x_test);

%% Define the RBF Kernel Function
sigma_f = 1.0; % signal variance
l = 1.0;       % lengthscale
kernel = @(x1, x2) sigma_f^2 * exp(- (x1 - x2).^2 / (2 * l^2));
%% Initialize GIF parameters
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
filename = fullfile(scriptDir, 'mogp_indep_demo.gif');
frame_count = 0; % gif frame

figure;
set(gcf,'Color', 'w', 'Position', [100, 100, 800, 600]);

for N = 5:N_total
    % Subset of data
    x = x_full(1:N);
    y = y_full(1:N);
    z = z_full(1:N);
    
    % Compute kernel between training points 
    K_xx = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K_xx(i, j) = kernel(x(i), x(j));
        end
    end
    
    % Add noise variance to the kernel
    K_y = K_xx + sigma_y^2 * eye(N); % for output y
    K_z = K_xx + sigma_z^2 * eye(N); % for output z
    
    % Compute kernel between training and test points
    K_xs = zeros(N, N_test);
    for i = 1:N
        for j = 1:N_test
            K_xs(i, j) = kernel(x(i), x_test(j));
        end
    end
    
    % Predicted mean and var for y
    mu_y = K_xs' * (K_y \ y);
    K_ss = zeros(N_test, N_test);
    for i = 1:N_test
        for j = 1:N_test
            K_ss(i, j) = kernel(x_test(i), x_test(j));
        end
    end
    var_y = diag(K_ss - K_xs' * (K_y \ K_xs));
    
    % Predicted mean and var for z
    mu_z = K_xs' * (K_z \ z);
    var_z = diag(K_ss - K_xs' * (K_z \ K_xs));
    
    % Plot the Results
    clf; % clear current figure window
    subplot(2, 1, 1);
    hold on;
    fill([x_test; flipud(x_test)], [mu_y + 2*sqrt(var_y); flipud(mu_y - 2*sqrt(var_y))], [7 7 7]/8, 'EdgeColor', 'none'); % CI for y
    % Predicted mean for y
    plot(x_test, mu_y, 'b-', 'LineWidth', 2);
    % Training data for y
    scatter(x, y, 25, 'r', 'filled');
    xlabel('x');
    ylabel('y');
    title(['Gaussian Process Regression for y (N = ' num2str(N) ')']);
    legend('95% CI', 'Predicted Mean', 'Training Data', 'Location', 'Best');
    hold off;
    
    subplot(2, 1, 2);
    hold on;
    % Confidence interval for z
    fill([x_test; flipud(x_test)], [mu_z + 2*sqrt(var_z); flipud(mu_z - 2*sqrt(var_z))], [7 7 7]/8, 'EdgeColor', 'none');
    % Predicted mean for z
    plot(x_test, mu_z, 'g-', 'LineWidth', 2);
    % Training data for z
    scatter(x, z, 25, 'r', 'filled');
    xlabel('x');
    ylabel('z');
    title(['Gaussian Process Regression for z (N = ' num2str(N) ')']);
    legend('95% CI', 'Predicted Mean', 'Training Data', 'Location', 'Best');
    hold off;
    
%% Generate gif
    drawnow;
    frame = getframe(gcf);
    img = frame2im(frame);
    [img_ind, cmap] = rgb2ind(img, 256);
    
    if frame_count == 0
        imwrite(img_ind, cmap, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1);
    else
        imwrite(img_ind, cmap, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
    end
    frame_count = frame_count + 1;

    pause(1);
end
