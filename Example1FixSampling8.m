% Clear previous variables and figures
clear;
close all;
%somgthing abt max singular value 

% Parameters
sigma1 = 1;
sigma2 = 2;
sigma3 = 3;
lambda1 = 0.1;
lambda2 = 0.2;
lambda3 = 0.3;
lambda4 = 0.4;
psi1 = 0;
psi2 = pi/4;

% Common length for each filter (17 points)
filter_length = 17;

% Sampling points
x1 = linspace(-3, 3, filter_length); % For sigma3 = 1
x2 = linspace(-6, 6, filter_length); % For sigma1 = 2
x3 = linspace(-9, 9, filter_length); % For sigma2 = 3

% Generate filters (cosine)
g1 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda1 + psi1);
g2 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / (lambda1 - 0.2) + psi2);
g3 = exp(-x1.^2 / (2 * (sigma1 - 0.5)^2)) .* cos(2 * pi * x1 / lambda2 + psi1);
g4 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda2 + psi2);
g5 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda3 + psi1);
g6 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda3 + psi2);
g7 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda4 + psi1);
g8 = exp(-x1.^2 / (2 * sigma1^2)) .* cos(2 * pi * x1 / lambda4 + psi2);

% Generate filters (sine)
h1 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda1 + psi1);
h2 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda1 + psi2);
h3 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda2 + psi1);
h4 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda2 + psi2);
h5 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda3 + psi1);
h6 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda3 + psi2);
h7 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda4 + psi1);
h8 = exp(-x2.^2 / (2 * sigma2^2)) .* sin(2 * pi * x2 / lambda4 + psi2);


h9 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi1);
h10 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi2);
h11 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi1);
h12 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi2);
h13 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi1);
h14 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi2);
h15 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi1);
h16 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi2);
h17 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi1);
h18 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi2);
h19 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi1);
h20 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi2);
h21 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi1);
h22 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi2);
h23 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi1);
h24 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi2);
h25 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi1);
h26 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda1 + psi2);
h27 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi1);
h28 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda2 + psi2);
h29 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi1);
h30 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda3 + psi2);
h31 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi1);
h32 = exp(-x3.^2 / (2 * sigma3^2)) .* sin(2 * pi * x3 / lambda4 + psi2);

% Combine filters into GaborMatrix
GaborMatrix = [g1; g2; g3; g4; g5; g6; g7; g8; h1; h2; h3; h4; h5; h6; h7; h8; h9; h10; h11; h12; h13; h14; h15; h16; h17; h18; h19; h20; h21; h22; h23; h24; h25; h26; h27; h28; h29; h30; h31; h32];
svdGaborMatrix = svd(GaborMatrix);
% Ensure each filter is zero mean
for i = 1:size(GaborMatrix, 1)
    GaborMatrix(i, :) = GaborMatrix(i, :) - mean(GaborMatrix(i, :));
end

% Check the sum of each row to ensure they are zero
Checkrow_sums = sum(GaborMatrix, 2);
disp('Sum of each row of GaborMatrix after adjustments:');
disp(Checkrow_sums);

% Define lowpass filter
lowpass = (1/2^16) * [1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1];

% Interpolate lowpass filter to the same length as Gabor filters
x_lowpass = linspace(-9, 9, length(lowpass));
lowpass_interpolated = interp1(x_lowpass, lowpass, x2, 'linear', 'extrap');

% Define matrix Q
Q_pre = [lowpass_interpolated; GaborMatrix];
SVDQ_pre = svd(Q_pre);

% Define matrix Q1
% Square root of each element of c
sqrt_c_inv = sqrt(lowpass_interpolated).^-1;

% Initialize the D1 matrix
num_rows = size(GaborMatrix, 1); % Number of rows in GaborMatrix
num_cols = filter_length;
D1 = zeros(num_rows, num_cols);

% Populate D1 with the product of each row of GaborMatrix and sqrt_c_inv
for i = 1:num_rows
    D1(i, :) = GaborMatrix(i, :) .* sqrt_c_inv;
end

Q_preNEW = [lowpass_interpolated; D1];
svdQ_preNEW = svd(Q_preNEW);

% Normalize the matrix Q by scaling with max singular value if needed
max_singular_value = max(svdQ_preNEW);
if max_singular_value > 1
    Q_preNEW = Q_preNEW / max_singular_value;
end

% Perform Singular Value Decomposition (SVD)
[U, S, V] = svd(Q_preNEW);
SVDQ = svd(Q_preNEW);

% Ensure the singular values are less than or equal to 1
singular_values = diag(S);

if all(singular_values <= 1)
    % Construct D2 matrix
    Sigma2 = diag(sqrt(1 - singular_values.^2));
    D2 = Sigma2 * V';
    
    % Form the Parseval frame
    Q2pre = [Q_preNEW; D2];
    
    % Make sure no imaginary
    Q2 = real(Q2pre);
    
    % Verify the Parseval frame
    if norm(Q2' * Q2 - eye(size(Q2, 2))) < 1e-10
        disp('Parseval frame verified.');
    else
        disp('Parseval frame verification failed.');
    end
else
    disp('Singular values are not all less than or equal to 1.');
    Q2 = []; % Handle the case where Q2 cannot be constructed
end

% If Q2 is successfully constructed, proceed with reconstruction
if ~isempty(Q2)
    % Define the function f
    N = 24000;
    x = linspace(-2, 2, N)'; % Define x range
    f = (x.^2 .* sin(x.^-2)) ./ exp(0.01 * x.^2);
    f(x == 0) = 0; % Handle division by zero

    % Display the function
    figure;
    plot(x, f);
    title('Function f(x) with a = 2');
    xlabel('x');
    ylabel('f(x)');

    % Initialize the approximation
    f_prime = zeros(size(f));

    % Lowpass filter reconstruction
    f_conv0 = conv(f, lowpass_interpolated, 'same'); % First convolution
    f_double_conv0 = conv(f_conv0, lowpass_interpolated, 'same'); % Second convolution
    f_prime = f_prime + f_double_conv0;

    % Perform double convolution with each orthogonalized highpass filter
    for i = 1:size(Q2, 1)
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
    scaling_factor = max(abs(f)) / max(abs(f_prime));
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

    % Calculate the reconstruction error
    ReconStructionErro = norm(f - f_prime) / norm(f);
    disp('Reconstruction Error:');
    disp(ReconStructionErro);
else
    disp('Q2 matrix could not be constructed due to singular values exceeding 1.');
end

%=======================================================================

% Plot the first 8 highpass filters
figure;
for i = 1:8
    subplot(4, 2, i);
    plot(x1, GaborMatrix(i, :));
    title(['Highpass Filter ', num2str(i)]);
    xlabel('x');
    ylabel('Amplitude');
end
sgtitle('First 8 Highpass Filters');

% Plot the next 8 highpass filters
figure;
for i = 9:16
    subplot(4, 2, i-8);
    plot(x2, GaborMatrix(i, :));
    title(['Highpass Filter ', num2str(i)]);
    xlabel('x');
    ylabel('Amplitude');
end
sgtitle('Next 8 Highpass Filters');

% Plot the third set of 8 highpass filters
figure;
for i = 17:24
    subplot(4, 2, i-16);
    plot(x3, GaborMatrix(i, :));
    title(['Highpass Filter ', num2str(i)]);
    xlabel('x');
    ylabel('Amplitude');
end
sgtitle('Third Set of 8 Highpass Filters');