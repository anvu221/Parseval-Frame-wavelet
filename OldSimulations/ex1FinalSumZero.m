function gaborBank = create1DGaborBank(numFilters, sigma, lambda, gamma, psi)
    % Create a 1D Gabor filter bank
    % numFilters: number of filters in the bank
    % sigma: standard deviation of the Gaussian envelope
    % lambda: wavelength of the sinusoidal factor
    % gamma: spatial aspect ratio
    % psi: phase offset of the sinusoidal factor
    
    gaborBank = cell(1, numFilters);
    for n = 1:numFilters
        theta = (n-1) * pi / numFilters; % Orientation of the filter
        gaborBank{n} = create1DGaborFilter(sigma, theta, lambda, gamma, psi);
    end
    
    % Concatenate the filters into a single matrix
    gaborBank = cat(1, gaborBank{:});
end

function gabor = create1DGaborFilter(sigma, theta, lambda, gamma, psi)
    % Create a 1D Gabor filter
    % sigma: standard deviation of the Gaussian envelope
    % theta: orientation of the filter
    % lambda: wavelength of the sinusoidal factor
    % gamma: spatial aspect ratio
    % psi: phase offset of the sinusoidal factor

    sz = fix(8 * sigma);
    if mod(sz, 2) == 0
        sz = sz + 1;
    end
    x = -fix(sz/2) : fix(sz/2);
    x_theta = x * cos(theta);
    y_theta = x * sin(theta);
    
    gb = exp(-0.5 * (x_theta.^2 + gamma^2 * y_theta.^2) / sigma^2) .* cos(2 * pi * x_theta / lambda + psi);
    gb = gb - mean(gb); % Subtract the mean from the coefficients to make them zero-mean
    gb(end) = gb(end) - sum(gb); % Adjust the last element to ensure the exact zero-sum condition
    
    gabor = gb;
end

function [reconstructionError, reconstructedSignal, filteredSignals, highpassSums] = filterAndReconstruct1DFunction(f, gaborBank, lowpass)
    % Filter the input function with the given filters and reconstruct it
    % f: input function
    % gaborBank: matrix containing the Gabor filters
    % lowpass: lowpass filter
    
    numFilters = size(gaborBank, 1);
    N = length(f);
    
    % Convolve with lowpass filter and normalize
    lowpass_filtered = conv(f, lowpass, 'same') / sum(lowpass);
    
    % Convolve with each highpass filter and normalize
    filteredSignals = zeros(N, numFilters);
    highpassSums = zeros(1, numFilters);
    for i = 1:numFilters
        filteredSignal = conv(f, gaborBank(i, :), 'same');
        filteredSignals(:, i) = filteredSignal / sum(abs(filteredSignal));
        highpassSums(i) = sum(gaborBank(i, :)); % Sum of the coefficients of each highpass filter
    end
    
    % Reconstruction using highpass and lowpass filters
    reconstructedSignal = sum(filteredSignals, 2) + lowpass_filtered;
    
    % Compute reconstruction error
    reconstructionError = norm(f - reconstructedSignal) / norm(f);
end

% Parameters
N = 90000; % Number of sample points
a = 2; % Example constant for the function f(x)

% Define the function f as a column vector in the interval [-2, 2]
x = linspace(-2, 2, N)';
% Define the function with handling for x = 0 to avoid division by zero
f = (x.^a .* sin(x.^-a)) ./ exp(0.01 * x);
f(x == 0) = 0; % Define the function at x = 0

% Create the Gabor bank and lowpass filter
numFilters = 16;
sigma = 2;
lambda = 4;
gamma = 1;
psi = 0;
gaborBank = create1DGaborBank(numFilters, sigma, lambda, gamma, psi);
lowpass = [0.0000    0.0002    0.0018    0.0085    0.0278    0.0667    0.1222    0.1746    0.1964    0.1746 ...
           0.1222    0.0667    0.0278    0.0085    0.0018    0.0002    0.0000];

% Add all elements of the lowpass filter together
lowpass_sum = sum(lowpass);

% Print the filters
disp('Lowpass filter:');
disp(lowpass);
for i = 1:numFilters
    disp(['Highpass filter ' num2str(i) ':']);
    disp(gaborBank(i, :));
end

% Filter and reconstruct the function
[reconstructionError, reconstructedSignal, filteredSignals, highpassSums] = filterAndReconstruct1DFunction(f, gaborBank, lowpass);

% Plotting the original function and the reconstructed function
figure;
subplot(2,1,1);
plot(x, f);
title('Original Function');
subplot(2,1,2);
plot(x, reconstructedSignal);
title('Reconstructed Function');

% Display reconstruction error and sum of lowpass filter
disp(['Reconstruction Error: ' num2str(reconstructionError)]);
disp(['Sum of Lowpass Filter: ' num2str(lowpass_sum)]);

% Add a text box below the plots with the reconstruction error
str = ['Reconstruction Error: ' num2str(reconstructionError)];
annotation('textbox', [0.4, 0.05, 0.2, 0.05], 'String', str, 'FontSize', 12, 'Color', 'red', 'BackgroundColor', 'white', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

% Display the sum of the coefficients for each highpass filter
for i = 1:numFilters
    disp(['Sum of coefficients for highpass filter ' num2str(i) ': ' num2str(highpassSums(i))]);
end

% Normalize the filter bank
BMatrix = [lowpass; gaborBank];
frameSum = sqrt(sum(BMatrix.^2, 1));
normalizedBMatrix = BMatrix ./ frameSum;

% Extract the normalized lowpass and gabor bank
lowpass_normalized = normalizedBMatrix(1, :);
gaborBank_normalized = normalizedBMatrix(2:end, :);

% Filter and reconstruct the function
[reconstructionError, reconstructedSignal, filteredSignals, highpassSums] = filterAndReconstruct1DFunction(f, gaborBank_normalized, lowpass_normalized);
