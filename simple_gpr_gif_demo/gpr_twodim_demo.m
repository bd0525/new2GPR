% Generate sample data
X = -3 + 6 * rand(20, 2); % 20 random 2D points
Y = sin(X(:,1)) .* cos(X(:,2)) + 0.05 * randn(20, 1); % Observations Y include some noise
n = length(X);

%% Define the RBF kernel function
sigma = 1.0; % Bandwidth of the RBF kernel
RBF_kernel = @(x1, x2) exp(-norm(x1 - x2)^2 / (2 * sigma^2));
%%
% Compute the full kernel matrix
K = zeros(n, n); % initialize the kernel matrix
for i = 1:n
    for j = 1:n
        K(i,j) = RBF_kernel(X(i,:), X(j,:));
    end
end

K = K + 0.05^2 * eye(n);

% Test inputs for prediction
[X1_test_grid, X2_test_grid] = meshgrid(linspace(-3, 3, 50), linspace(-3, 3, 50));
X_test = [X1_test_grid(:), X2_test_grid(:)];
m_test = size(X_test, 1);

% Kernel between test points and training points
K_test = zeros(m_test, n);
for k = 1:m_test
    for j = 1:n
        K_test(k,j) = RBF_kernel(X_test(k,:),X(j,:));
    end
end

% Precompute diagonal of K_test_test / For RBF exp(0)=1
k_star_star = ones(m_test, 1);

figure;
set(gcf, 'color', 'w'); % gcf stands for 'get current figure'

scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
filename = fullfile(scriptDir, 'gpr_twodim_demo.gif');

for i = 1:n
    % Get relevant submatrices
    K_i = K(1:i, 1:i);
    K_inv = inv(K_i);
    K_test_i = K_test(:,1:i);
    Y_i = Y(1:i);

    % Compute predictive mean
    mu_test = K_test_i * K_inv * Y_i;
    
    % Compute predictive variance
    var_test = zeros(m_test, 1);
    for k = 1:m_test
        k_star = K_test_i(k,:)';
        var_test(k) = k_star_star(k) - k_star' * K_inv * k_star;
    end
    std_test = sqrt(var_test);
    
    %% Plot the results
    cla; % clear current axes
    hold on;
    
    %Plot the mean surface[X1,X2] with std as color
    mu_test_grid = reshape(mu_test,size(X1_test_grid));
    std_test_grid = reshape(std_test,size(X1_test_grid));
    surf(X1_test_grid, X2_test_grid, mu_test_grid, std_test_grid, 'EdgeColor','none');
    colorbar;
    hcb = colorbar;
    ylabel(hcb, 'Standard Deviation');
    
    % Plot the training points
    scatter(X(1:i,1), X(1:i,2), 50, 'filled', 'MarkerFaceColor', 'red');

    hold off;
    legend('GPR mean surface', 'Training points', 'Location', 'Best');
    xlabel('X1');
    ylabel('X2');
    zlabel('Y');
    title(['Gaussian Process Regression (N = ' num2str(i) ')']);
    view(3);
    %% Generate gif
    drawnow;
    frame = getframe(gcf); % Capture figure frame
    im = frame2im(frame); % Convert frame to image
    [A, map] = rgb2ind(im, 256); % Convert image to indexed image
    if i == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
    end

    pause(1);
end
