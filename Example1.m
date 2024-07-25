%othogonal
%Persevalization
%Try adjust the lambda and do the long filters
%I notice no change is the result

%Chapter 1 create GaborFilters
clear;
% Parameters
sigma1 = 2;
sigma2 = 3;
sigma3 = 1;
lambda1 = 0.1;
lambda2 = 0.2;
lambda3 = 0.3;
lambda4 = 0.4;
psi1 = 0;
psi2 = pi/4;

% Sampling points
%x = linspace(-8, 8, 17); % Changed from [-2, 2] to [-6, 6]
x = -8:8;%new sampling method
% Generate filters (cosine)
g1 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda1 + psi1);
g2 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda1 + psi2);
g2prime = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / (lambda1-0.2) + psi2);
g3 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda2 + psi1);
g3prime = exp(-x.^2 / (2 * (sigma1-0.5)^2)) .* cos(2 * pi * x / lambda2 + psi1);
g4 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda2 + psi2);
g5 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda3 + psi1);
g6 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda3 + psi2);
g6prime = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / (lambda3) + psi2);
g7 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda4 + psi1);
g8 = exp(-x.^2 / (2 * sigma1^2)) .* cos(2 * pi * x / lambda4 + psi2);

% Generate filters (sine)
h1 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda1 + psi1);
h2 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda1 + psi2);
h3 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda2 + psi1);
h4 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda2 + psi2);
h5 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda3 + psi1);
h6 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda3 + psi2);
h7 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda4 + psi1);
h8 = exp(-x.^2 / (2 * sigma2^2)) .* sin(2 * pi * x / lambda4 + psi2);
h9 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda1 + psi1);
h10 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda1 + psi2);
h11 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda2 + psi1);
h12 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda2 + psi2);
h13 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda3 + psi1);
h14 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda3 + psi2);
h15 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda4 + psi1);
h16 = exp(-x.^2 / (2 * sigma1^2)) .* sin(2 * pi * x / lambda4 + psi2);
h17 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda1 + psi1);
h18 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda1 + psi2);
h19 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda2 + psi1);
h20 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda2 + psi2);
h21 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda3 + psi1);
h22 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda3 + psi2);
h23 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda4 + psi1);
h24 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda4 + psi2);
h25 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda1 + psi1);
h26 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda1 + psi2);
h27 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda2 + psi1);
h28 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda2 + psi2);
h29 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda3 + psi1);
h30 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda3 + psi2);
h31 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda4 + psi1);
h32 = exp(-x.^2 / (2 * sigma3^2)) .* sin(2 * pi * x / lambda4 + psi2);

% Plot filters (cosine)
figure;
subplot(4, 2, 1); plot(x, g1); title('Filter 1: \sigma=2, \lambda=1, \psi=0 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 2); plot(x, g2); title('Filter 2: \sigma=2, \lambda=1, \psi=\pi/2 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 3); plot(x, g3); title('Filter 3: \sigma=2, \lambda=2, \psi=0 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 4); plot(x, g4); title('Filter 4: \sigma=2, \lambda=2, \psi=\pi/2 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 5); plot(x, g5); title('Filter 5: \sigma=2, \lambda=3, \psi=0 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 6); plot(x, g6); title('Filter 6: \sigma=2, \lambda=3, \psi=\pi/2 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 7); plot(x, g7); title('Filter 7: \sigma=2, \lambda=4, \psi=0 (cos)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 8); plot(x, g8); title('Filter 8: \sigma=2, \lambda=4, \psi=\pi/2 (cos)'); xlabel('x'); ylabel('Amplitude');
sgtitle('Highpass Gabor Filters (Cosine)');

% Plot filters (sine)
figure;
subplot(4, 2, 1); plot(x, h1); title('Filter 9: \sigma=3, \lambda=1, \psi=0 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 2); plot(x, h2); title('Filter 10: \sigma=3, \lambda=1, \psi=\pi/2 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 3); plot(x, h3); title('Filter 11: \sigma=3, \lambda=2, \psi=0 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 4); plot(x, h4); title('Filter 12: \sigma=3, \lambda=2, \psi=\pi/2 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 5); plot(x, h5); title('Filter 13: \sigma=3, \lambda=3, \psi=0 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 6); plot(x, h6); title('Filter 14: \sigma=3, \lambda=3, \psi=\pi/2 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 7); plot(x, h7); title('Filter 15: \sigma=3, \lambda=4, \psi=0 (sin)'); xlabel('x'); ylabel('Amplitude');
subplot(4, 2, 8); plot(x, h8); title('Filter 16: \sigma=3, \lambda=4, \psi=\pi/2 (sin)'); xlabel('x'); ylabel('Amplitude');
sgtitle('Highpass Gabor Filters (Sine)');


% Combine filters into gaborMatrix
%Chapter2 Try to eliminate bad filters which makes linearly dependent
GaborMatrix = [g1; g2prime; g3prime; g4; g5; g6prime; g7; g8; h1; h2; h3; h4; h5; h6; h7; h8; h9; h10; h11; h12; h13; h14; h15; h16; h17; h18; h19; h20; h21; h22; h23; h24; h25; h26; h27; h28; h29; h30; h31; h32];
svdGaborMatrix = svd(GaborMatrix);

%Chapter3 Subtract the mean to make each filter zero mean
for i = 1:size(GaborMatrix, 1)
    GaborMatrix(i, :) = GaborMatrix(i, :) - mean(GaborMatrix(i, :));
    GaborMatrix(i, end) = GaborMatrix(i, end) - sum(GaborMatrix(i, :));
end

% Check the sum of each row to ensure they are zero
Checkrow_sums = sum(GaborMatrix, 2);
disp('Sum of each row of GaborMatrix2 after adjustments:');
disp(Checkrow_sums);

%Chapter 4: add the lowpass and the function of interest
lowpass = (1/2^16) * [1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1];
%Define matrix Q
Q_pre = [lowpass; GaborMatrix];

SVDQ_pre = svd(Q_pre);

%Define matrix Q1
% Square root of each element of c
sqrt_c_inv = sqrt(lowpass).^-1;

% Initialize the D1 matrix
% Number of rows and columns
num_rows = 40;
num_cols = 17;
D1 = zeros(num_rows, num_cols);


% Populate D1 with the product of each row of GaborFilters and sqrt_c_inv
for i = 1:num_rows
    D1(i, :) = GaborMatrix(i, :) .* sqrt_c_inv;
end

Q_preNEW = [lowpass;D1];
svdQ_preNEW = svd(Q_preNEW);

% Define the function (with handling for x = 0 to avoid division by zero)
% Define the function f as a column vector in the interval [-2, 2],
% breaking into 4/6000 small subintervals.
N = 24000;
x = linspace(-2, 2, N)'; % The apostrophe (') here transposes the row vector to a column vector

% Function:
f = (x.^2 .* sin(x.^-2)) ./ exp(0.01 * x.^2);
f(x == 0) = 0; % Define the function at x = 0 to avoid division by zero
sizef = size(f);

% Display the function
figure;
plot(x, f);
title('Function f(x) with a = 2');
xlabel('x');
ylabel('f(x)');

% Initialize the approximation
f_prime = zeros(size(f));


% Make highpass filters orthogonal to lowpass filter
for i = 1:size(GaborMatrix, 1)
    highpass_filter = GaborMatrix(i, :);
    projection = (dot(lowpass, highpass_filter) / dot(lowpass, lowpass)) * lowpass;
    GaborMatrix(i, :) = highpass_filter - projection;
end

%Normalize by max singular value
%Q = Q_pre / 8.8; %this is old method (projection to c)
Q = Q_preNEW / max(svd(Q_preNEW)); %this is new method (define D1)

% Perform Singular Value Decomposition (SVD)
[U, S, V] = svd(Q);
SVDQ = svd(Q);
% Ensure the singular values are less than or equal to 1
singular_values = diag(S);

if all(singular_values <= 1.0005)
    % Construct D2 matrix
    Sigma2 = diag(sqrt(1 - singular_values.^2));
    D2 = Sigma2 * V';
    
    % Form the Parseval frame
    Q2pre = [Q; D2];
    
    % make sure no imaginary
    Q2 = real(Q2pre);
    
    % Verify the Parseval frame
    if norm(Q2' * Q2 - eye(size(Q2, 2))) < 1e-10
        disp('Parseval frame verified.');
    else
        disp('Parseval frame verification failed.');
    end
else
    disp('Singular values are not all less than or equal to 1.');
end

% lowpass filter reconstruction
f_conv0 = conv(f, lowpass, 'same'); % First convolution
f_double_conv0 = conv(f_conv0, lowpass, 'same'); % Second convolution
f_prime = f_prime + f_double_conv0;

% Perform double convolution with each orthogonalized highpass filter
for i = 1:size(Q2, 1) %Replace GaborMatrix2 with Q2
    H = Q2(i, :);
    f_conv = conv(f, H, 'same'); % First convolution
    f_double_conv = conv(f_conv, H, 'same'); % Second convolution
    f_prime = f_prime + f_double_conv;
end

% Calculate the error
error = f - f_prime;

% Display the reconstructed function
figure;
plot(x, f_prime, 'r--', 'DisplayName', 'Reconstructed Function $\hat{f}(x)$');
title('Reconstructed Function');
xlabel('x');
ylabel('Function Value');
legend('Interpreter', 'latex');

% Trick to scale
% Calculate the scaling factor
scaling_factor = max(abs(f)) / max(abs(f_prime));
% Apply the scaling factor to the reconstructed signal
f_prime = f_prime * scaling_factor;

% Display the original and reconstructed function
figure;
plot(x, f, 'b', 'DisplayName', 'Original Function f(x)');
hold on;
plot(x, f_prime, 'r--', 'DisplayName', 'Reconstructed Function $\hat{f}(x)$');
title('Original and Reconstructed Function');
xlabel('x');
ylabel('Function Value');
legend('Interpreter', 'latex');

% Calculate the error
%error = f - f_prime;

% Calculate the Mean Squared Error (MSE)
%mse = mean(error.^2);

% Calculate the Peak Signal-to-Noise Ratio (PSNR)
%psnr_value = 10 * log10(max(f.^2) / mse);


ReconStructionErro = norm(f-f_prime)/norm(f);