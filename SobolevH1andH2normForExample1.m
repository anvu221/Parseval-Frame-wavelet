% Define the function parameter a
a = 2;

% Define the function and new domain
N = 2^12;
x = linspace(-10, 10, N);
h = (x(end) - x(1)) / (N - 1);

% Define the function f(x) = x^a * sin(x^-a) with special handling at x = 0
f = arrayfun(@(xi) (xi^a * sin(xi^-a)) ./ exp(0.01 * xi), x);
f(x == 0) = 0;  % Define f(0) = 0 as a special case

% Compute the L2 norm of f
L2_norm_f = sqrt(sum(f.^2) * h);

% Compute the first derivative using central differences
f_prime = zeros(1, N-2);
for i = 2:N-1
    if x(i) ~= 0
        f_prime(i-1) = (f(i+1) - f(i-1)) / (2 * h);
    else
        f_prime(i-1) = 0;  % Special handling for x = 0
    end
end

% Compute the L2 norm of the first derivative
L2_norm_f_prime = sqrt(sum(f_prime.^2) * h);

% Compute the second derivative using central differences
f_double_prime = zeros(1, N-2);
for i = 2:N-1
    if x(i) ~= 0
        f_double_prime(i-1) = (f(i+1) - 2*f(i) + f(i-1)) / (h^2);
    else
        f_double_prime(i-1) = 0;  % Special handling for x = 0
    end
end

% Compute the L2 norm of the second derivative
L2_norm_f_double_prime = sqrt(sum(f_double_prime.^2) * h);

% Compute the H1 norm
H1_norm = sqrt(L2_norm_f^2 + L2_norm_f_prime^2);

% Compute the H2 norm
H2_norm = sqrt(L2_norm_f^2 + L2_norm_f_prime^2 + L2_norm_f_double_prime^2);

% Display the results
fprintf('H1 norm of f: %.5f\n', H1_norm);
fprintf('H2 norm of f: %.5f\n', H2_norm);
