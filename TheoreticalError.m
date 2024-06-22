% Given values
a = 1;
N = 3;
s = 2;
sigma = 1;

% Define M_nu function
M_nu = @(nu) max(1, log2(0.5 + abs(nu)));

% Calculate A
A = a^2 * (2 + sum(arrayfun(@(l) 2^N / l^N, 5:2:1000)));  % Approximating with a large sum

% Calculate C(s)
xi_range = linspace(-0.5, 0.5, 100);
C_s = max(arrayfun(@(xi) sum(arrayfun(@(nu) 1 / (1 + (xi + nu)^2)^(s/2), -100:100)), xi_range));  % Approximating with a large range

% Calculate B(s)
B_s = max(arrayfun(@(xi) sum(arrayfun(@(nu) (M_nu(nu) + (N + 1) * pi) / (1 + (xi + nu)^2)^(s/2), [-100:-2, 2:100])), xi_range));  % Approximating with a large range

% Main expression calculation
result = sigma + 2 * sigma * A * (C_s + 3 + (9 * pi / 8) * (N + 1)) + 2 * sigma * A * B_s;

% Display the result
disp(result);