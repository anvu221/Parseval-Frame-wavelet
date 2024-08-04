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
    gabor = gb;
end

% Example usage
numFilters = 16;
sigma = 2;
lambda = 3;
gamma = 1;
psi = 0;

gaborBank = create1DGaborBank(numFilters, sigma, lambda, gamma, psi);

% Plotting the Gabor filters in two figures with 8 filters each
figure;
for i = 1:8
    subplot(8, 1, i);
    plot(gaborBank(i, :)); % Accessing each filter from the matrix
    title(['Gabor Filter ' num2str(i)]);
end

figure;
for i = 9:16
    subplot(8, 1, i-8);
    plot(gaborBank(i, :)); % Accessing each filter from the matrix
    title(['Gabor Filter ' num2str(i)]);
end
