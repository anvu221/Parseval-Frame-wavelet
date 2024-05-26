% Parameters
N = 500; % Number of sample points
M = 50; % Number of frequency channels
L = 50; % Window length
sigma = L / 6; % Standard deviation of the Gaussian window
a = 2; % Example constant for the function f(x)

% Define the function f as a column vector in the interval [-2, 2]
x = linspace(-2, 2, N)'; 

% Define the function with handling for x = 0 to avoid division by zero
f = (x.^a .* sin(x.^-a)) ./ exp(0.01 * x);
f(x == 0) = 0; % Define the function at x = 0

% Define the Gaussian window
window = @(n) exp(-0.5 * ((n - (L-1)/2) / sigma).^2);

% Generate the Gabor frame
G = zeros(N, M*N);
for k = 0:M-1
    for n = 0:N-1
        idx = mod((0:L-1) + n, N) + 1;
        gabor_element = window(0:L-1)' .* exp(2*pi*1i*k*(0:L-1)'/L);
        G(idx, k*N + n + 1) = G(idx, k*N + n + 1) + gabor_element;
    end
end

% Normalize the entire Gabor frame
G = G / norm(G, 'fro');

% Project the function f onto the Gabor frame
coefficients = G' * f;

% Reconstruct the function from the Gabor frame
f_reconstructed = real(G * coefficients);

% Calculate the reconstruction error
reconstruction_error = norm(f - f_reconstructed) / norm(f);

% Display the results
disp('Reconstruction error:');
disp(reconstruction_error);

% Plot the original and reconstructed functions
figure;
subplot(2, 1, 1);
plot(x, f);
title('Original Function');
subplot(2, 1, 2);
plot(x, f_reconstructed);
title('Reconstructed Function');

% Add reconstruction error to the plot
annotation('textbox', [0.15,0.05,0.3,0.05], 'String', ['Reconstruction error: ', num2str(reconstruction_error)], 'LineStyle', 'none', 'HorizontalAlignment', 'center', 'FontSize', 10);