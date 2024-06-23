% Given variables
a = 1;
N = 16;
s = 2;
sigma = 0.114;

% Calculate A
A = 2; % Initial value
for k = 2:100 % Sum up to a large number to approximate the series
    A = A + 2^N / (2*k+1)^N;
end

% Calculate C(s)
C_s = 0;
for nu = -100:100 % Sum from -100 to 100 to approximate the series
    C_s = C_s + 1 / (1 + nu^2);
end
C_s = max(1, C_s); % The maximum value within the range [-1/2, 1/2] is the same as at 0

% Calculate B(s)
B_s = 0;
for nu = 2:100 % Sum up to a large number to approximate the series
    M_nu = max(1, log2(0.5 + abs(nu)));
    B_s = B_s + (M_nu + (N + 1) * pi) / (1 + nu^2);
end

% Combine everything
term1 = sigma;
term2 = 2 * sigma * A * (C_s + 3 + (9 * pi / 8) * (N + 1));
term3 = 2 * sigma * A * B_s;

result = term1 + term2 + term3;

% Display the result
disp(['Result: ', num2str(result)]);
