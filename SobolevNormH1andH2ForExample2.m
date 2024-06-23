% Define the function parameter a
a = 2;

% Define the function and new domain
N = 2^12;
x = linspace(-10, 10, N);
h = (x(end) - x(1)) / (N - 1);

% Define the function f(x) based on the given piecewise definition
f = zeros(size(x));
for i = 1:length(x)
    if abs(x(i)) <= 1
        f(i) = 1 - abs(x(i))^a;
    else
        f(i) = 0;
    end
end

% Compute the L2 norm of f
L2_norm_f = sqrt(sum(f.^2) * h);

% Compute the first derivative using central differences
f_prime = zeros(1, N-2);
for i = 2:N-1
    f_prime(i-1) = (f(i+1) - f(i-1)) / (2 * h);
end

% Compute the L2 norm of the first derivative
L2_norm_f_prime = sqrt(sum(f_prime.^2) * h);

% Compute the second derivative using central differences
f_double_prime = zeros(1, N-2);
for i = 2:N-1
    f_double_prime(i-1) = (f(i+1) - 2*f(i) + f(i-1)) / (h^2);
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
