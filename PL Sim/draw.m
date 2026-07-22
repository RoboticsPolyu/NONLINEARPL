clc;
close all;

% Number of anchors
numAnchors = 8;

% Anchor positions (x, y, z)
anchorPos = [0, 0, 0; 10, 0, 0; 10, 10, 0; 0, 10, 0; 5, 0, 5; 5, 10, 5; 0, 5, 5; 10, 5, 5];

% Simulated moving anchor position (x, y, z)
t = linspace(0, 2*pi, 100)';
movingAnchorPos = [5 + cos(t), 5 + sin(t), 5 + sin(2*t)];

% Standard deviations for Gaussian noise and bias
gaussianStd = 0.05; % Standard deviation for Gaussian noise
sigma = 0.3; % Bias noise
alpha = 0.58;

% assuming tow anchor's bias <= 0.3
% Simulate distance measurements with noise
distances = zeros(100, numAnchors);
distances2 = zeros(100, numAnchors);
Jacobian = zeros(numAnchors, 3);
Q = eye(numAnchors)*gaussianStd*gaussianStd;
W = inv(Q);

for i = 1:100
    if i <= 50
        
        for j = 1:numAnchors
            % True distance
            trueDistance = norm(movingAnchorPos(i, :) - anchorPos(j, :));
            
            % Add Gaussian noise and bias noise
            if j >= 4
                bias = 0.25;
                noise = gaussianStd * randn + bias;
            % elseif j == 1
            %     bias = 0.20;
            %     noise = gaussianStd * randn + bias;
            else 
                noise = gaussianStd * randn;
            end
            distances(i, j) = trueDistance + noise;
        end
    else
        bias = - 0.0;
        for j = 1:numAnchors
            % True distance
            trueDistance = norm(movingAnchorPos(i, :) - anchorPos(j, :));
            
            % Add Gaussian noise and bias noise
            if j == 6 || j == 2
                noise = gaussianStd * randn + bias;
            else 
                noise = gaussianStd * randn;
            end
            distances(i, j) = trueDistance + noise;
        end
    end
end

% Plot results
figure;
plot3(movingAnchorPos(:, 1), movingAnchorPos(:, 2), movingAnchorPos(:, 3), 'k', 'LineWidth', 1);
% hold on;
% scatter3(anchorPos(:, 1), anchorPos(:, 2), anchorPos(:, 3), 'filled', 'k');
% legend('True Position', 'Estimated Position (NLO)', 'Anchor Positions');


% Estimate position using Nonlinear Optimization with fminunc
estimatedPos_NLO = zeros(100, 3);
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');

for i = 1:100
    syms x y z;
    f = sum((sqrt(sum((anchorPos - [x y z]).^2, 2)) - distances(i, :)').^2) - alpha*alpha;
    objectiveFunc = @(pos) sum((sqrt(sum((anchorPos - pos).^2, 2)) - distances(i, :)').^2);
    interval = [-10 10 -10 10 0 15];
    hold on;
    fimplicit3(f,interval,'EdgeColor','none','FaceAlpha',.1);
    initialGuess = [0, 0, 0];
    estimatedPos_NLO(i, :) = fminunc(objectiveFunc, initialGuess, options);
end

Omega = zeros(100, 3);
Chi_test = zeros(100, 1);
level = 0.1;
m = numAnchors;
chi2_critical_value = zeros(100, 1);
df = zeros(100, 1);
hold on;
plot3(estimatedPos_NLO(:, 1), estimatedPos_NLO(:, 2), estimatedPos_NLO(:, 3), 'r-.', 'LineWidth', 2);

grid on;
xlabel('X Position');
ylabel('Y Position');
zlabel('Z Position');
title('True and Estimated Positions of Moving Anchor with Noise and Anchor Coordinates');
grid on;

figure;
for i = 1:100
    % for k = 4:numAnchors
        % m = numAnchors + 4 - k;
        m = numAnchors;
        Jacobian = zeros(m, 3);
        Q = eye(m)*gaussianStd*gaussianStd;
        W = inv(Q);
        iteration = 20;

        % for index = 1:iteration
            selected_indices = randperm(numAnchors, m);
            sel_anchorPos = anchorPos(selected_indices,:);
            dist = distances(i, :)';
            sel_distances = dist(selected_indices);
    
            relative_pos = sel_anchorPos - estimatedPos_NLO(i, :);
            di = sqrt(sum((relative_pos).^2, 2));
            Jacobian = relative_pos ./ di;
            Omega(i,:) = 3* sqrt(diag(inv(Jacobian'*W*Jacobian)));
            test = W*(eye(m) - Jacobian*inv(Jacobian'*W*Jacobian)*Jacobian'*W);
            ri = di - sel_distances;
            hold on; plot(ri);
            Chi_test(i,1) = ri'* test *ri;

            % df(i,1) = m - 3;
            % if(df(i,1) < 5)
            %     print df
            % end
            % chi2_critical_value(i,1) = chi2inv(1 - level, df(i,1));
            % if Chi_test(i,1) < chi2_critical_value(i,1)
            %     break;
            % end
        % end
    %     if Chi_test(i,1) < chi2_critical_value(i,1)
    %         syms x y z;
    %         f = (sqrt(sum(([x y z] - sel_anchorPos).^2, 2)) - sel_distances)' * test * (sqrt(sum(([x y z] - sel_anchorPos).^2, 2)) - sel_distances) - chi2_critical_value(i,1);
    %         % f = sum((sqrt(sum( ( (sel_anchorPos - [x y z])' * test.*(sel_anchorPos - [x y z]), 2)) - sel_distances(i, :)').^2) - chi2_critical_value(i,1); % (sel_anchorPos - [x y z])'* test * (sel_anchorPos - [x y z]) - sel_distances(i, :)').^2) - alpha*alpha;
    %         interval = [-10 10 -10 10 0 15];
    %         hold on;
    %         fimplicit3(f,interval,'EdgeColor','none','FaceAlpha',.3);
    %         break;
    %     end
    % end
end

% dx = inv(Jacobian' *W * Jacobian) * Jacobian' * W * [0. 0.25 0 0 0 0]';

% Compute estimation errors
estimationErrors_NLO = sqrt(sum((movingAnchorPos - estimatedPos_NLO).^2, 2));

% Calculate estimation errors
errors_nl = estimatedPos_NLO - movingAnchorPos;

figure; plot(Chi_test, 'r', 'LineWidth', 2 ); 
xlabel('Sample Number');
ylabel('Chi-square');
title('Chi-square statistic');
grid on;
hold on, plot(chi2_critical_value, 'b', 'LineWidth',2);
hold on; plot(df, 'g', 'LineWidth',2);
legend(['Chi_square', 'chi2_critical_value', 'df'])

% Plotting
figure;
plot(movingAnchorPos(:, 1), movingAnchorPos(:, 2), 'r', 'LineWidth', 2);
hold on;
plot(estimatedPos_NLO(:, 1), estimatedPos_NLO(:, 2), 'g-.', 'LineWidth', 2);

legend('True Position', 'Estimated Position (NLO)');
xlabel('X Position');
ylabel('Y Position');
title('True and Estimated Positions of Moving Anchor with Noise and Anchor Coordinates');
grid on;

figure
plot(abs(errors_nl(:, 1)), 'r--', 'DisplayName', 'NL X Error', 'LineWidth', 1.5); hold on;
plot(abs(errors_nl(:, 2)), 'g--', 'DisplayName', 'NL Y Error', 'LineWidth', 1.5);
plot(abs(errors_nl(:, 3)), 'b--', 'DisplayName', 'NL Z Error', 'LineWidth', 1.5);
xlabel('Sample Number');
ylabel('Error (m)');
title('Position Estimation Error using Nonlinear Optimization');
legend;
grid on;
ylim([0 0.6])
estimatedPos_PL = zeros(100, 3);
optimizationValues = zeros(100, 1); 
point_num = 1000;
PLO = zeros(point_num*100, 3);
% PL_Value = zeros(point_num*100, 1);

% for i = 51:100
%     % optimizationValues_b = 1000;
%     % for genj = 1 : 1000
%     %     % Number of dimensions
%     %     n = 8;
%     % 
%     %     % Generate a random 8D vector
%     %     b = randn(n, 1);
%     % 
%     %     % Scale the vector if necessary to satisfy the constraint
%     %     if b' * b > 0.09
%     %         b = b / sqrt(b' * b) * sqrt(0.09);
%     %     end
% 
% 
%         distances(i, 1) = distances(i, 1) - 0.22;
%         distances(i, 4) = distances(i, 4) - 0.22;
%         % distances(i, 7) = distances(i, 7) - 0.17;
% 
%         % distances2(i, :) = distances(i, :) - b';
% 
%         objectiveFunc = @(PL) sum((sqrt(sum((anchorPos - estimatedPos_NLO(i,:) - PL).^2, 2)) - distances(i, :)').^2);
%         initialGuess = [0, 0, 0];
%         [PLL, value] = fminunc(objectiveFunc, initialGuess, options);
%     % 
%     %     if value < optimizationValues_b
%             estimatedPos_PL(i, :) = PLL;
%             optimizationValues(i) = value;
%     %     end
%     % 
%     %     optimizationValues_b = value;
%     % end
% end
for i = 1:100
    optimizationValues(i) = 1000;
    estimatedPos_PL(i, :) = 0;
    for genj = 1 : point_num
        % Number of dimensions
        n = 8;

        % Initialize the vector with zeros
        b = zeros(n, 1);

        % Randomly select two indices
        indices = randperm(n, 2);
        
        alpha = 0.10;
        % Generate two random elements such that their squared sum is <= 0.09
        magnitude = sqrt(alpha);
        elements = randn(2, 1);

        % Scale the elements to ensure the constraint is met
        if sum(elements.^2) > alpha
            elements = elements / sqrt(sum(elements.^2)) * magnitude;
        end

        % Assign the non-zero elements to the randomly selected indices
        b(indices) = elements / randi([1 100]);

        % % Display the generated vector
        % disp('Generated vector b:');
        % disp(b);
        % 
        % % Verify the constraint
        % disp('b^T * b:');
        % disp(b' * b);

        distances2(i, :) = distances(i, :) - b';

        objectiveFunc = @(PL) sum((sqrt(sum((anchorPos - estimatedPos_NLO(i,:) - PL).^2, 2)) - distances2(i, :)').^2);
        initialGuess = [0, 0, 0];
        [PLL, value] = fminunc(objectiveFunc, initialGuess, options);
        PLO(i*point_num + genj,:) = PLL;


        % if value < optimizationValues(i)
        if abs(PLL(1)) > estimatedPos_PL(i, 1)
            estimatedPos_PL(i, 1) = abs(PLL(1));
            optimizationValues(i) = value;
        end

        if abs(PLL(2)) > estimatedPos_PL(i, 2)
            estimatedPos_PL(i, 2) = abs(PLL(2));
            optimizationValues(i) = value;
        end

        if abs(PLL(3)) > estimatedPos_PL(i, 3)
            estimatedPos_PL(i, 3) = abs(PLL(3));
            optimizationValues(i) = value;
        end
            % optimizationValues(i) = value;
        % end

    end
end

% figure;
plot(abs(estimatedPos_PL(:, 1)) + Omega(:,1), 'r-', 'DisplayName', 'PL X ', 'LineWidth', 2); hold on;
plot(abs(estimatedPos_PL(:, 2)) + Omega(:,2), 'g-', 'DisplayName', 'PL Y ', 'LineWidth', 2);
plot(abs(estimatedPos_PL(:, 3)) + Omega(:,3), 'b-', 'DisplayName', 'PL Z ', 'LineWidth', 2);
plot(Omega(:,1), 'r-.', 'DisplayName', '3sigma X ', 'LineWidth', 2); hold on;
plot(Omega(:,2), 'g-.', 'DisplayName', '3sigma Y ', 'LineWidth', 2);
plot(Omega(:,3), 'b-.', 'DisplayName', '3sigma Z ', 'LineWidth', 2);

xlabel('Sample Number');
ylabel('length (m)');
title('PL');
legend;
grid on;

figure;
plot(PLO(:,1), PLO(:,2), 'g.');
figure;
plot(PLO(:,1), PLO(:,3), 'g.');

% figure;
% plot(optimizationValues, 'b-', 'LineWidth', 2);
% xlabel('Time Step');
% ylabel('Optimization Value');
% title('Optimization Values Over Time (NLO)');
% grid on;


% estimatedPos_PL_bias = zeros(100, 11);
% % Define the inequality constraint
% function [c, ceq] = constraints(PL)
%     c = (PL(4)^2 + PL(5)^2+PL(6)^2 +PL(7)^2 +PL(8)^2 +PL(9)^2 +PL(10)^2 +PL(11)^2)  - 0.3*0.3;
%     ceq = [];
% end
% 
% % Initial guess
% x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; % Initial guess for x1, x2, x3
% 
% for i = 51:100
%     % Define the objective function
%     h = @(PL_bias)sum((sqrt(sum((anchorPos - estimatedPos_NLO(i,:) - PL_bias(1:3)).^2, 2)) - distances(i, :)' - PL_bias(4:11)).^2); % Replace 'yourFunction' with your actual function
% 
%     % Call fmincon to solve the optimization problem
%     options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter'); % Display iterations
%     [x_opt, fval] = fmincon(@(x) h(x), x0, [], [], [], [], [], [], @constraints, options);
%     estimatedPos_PL_bias(i,:) = x_opt;
% end
% 
% figure;
% plot(abs(estimatedPos_PL_bias(:, 1)), 'r-', 'DisplayName', 'PL X ', 'LineWidth', 2); hold on;
% plot(abs(estimatedPos_PL_bias(:, 2)), 'g-', 'DisplayName', 'PL Y ', 'LineWidth', 2);
% plot(abs(estimatedPos_PL_bias(:, 3)), 'b-', 'DisplayName', 'PL Z ', 'LineWidth', 2);
% xlabel('Sample Number');
% ylabel('length (m)');
% title('PL');
% legend;
% grid on;