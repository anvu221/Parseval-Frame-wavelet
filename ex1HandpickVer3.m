% Parameters
N = 500; % Number of sample points
a = 2; % Example constant for the function f(x)

% Define the function f as a column vector in the interval [-2, 2]
x = linspace(-2, 2, N)'; 

% Define the function with handling for x = 0 to avoid division by zero
f = (x.^a .* sin(x.^-a)) ./ exp(0.01 * x);
f(x == 0) = 0; % Define the function at x = 0

% Define the custom frame matrix
frame_matrix = 10^-2 * [ -17.7  0       0       0       0       0       0       0       17.7;
               0    -25      0       0       0       0       0       25      0;
               0     0     -17.7     0       0       0       17.7    0       0;
               0     0       0     -25       0       25      0       0       0;
               0     0       0      -6.63    13.26   -6.63   0       0       0;
               0     0     -11.75    0       23.5    0      -11.75   0       0;
              -12.65 0       0       0       5       0       0       0     -12.65;
               0.002 0.001   0.0003  0.001  -0.008   0.0003  0.001   0.002   0.0003;
              -8.52  0.0288  9.59    0.233  -2.66    0.233   9.59  -8.52    0.0288;
               5.46 -0.939   5.69   -19      17.5   -19      5.69   5.46    -0.939;
               3.39 -21.5    3.4     8.1     13.2    8.1     3.4   -21.5     3.39];

% Ensure the frame matrix has correct dimensions
[frame_rows, frame_cols] = size(frame_matrix);

% Resize the frame matrix to match the length of f if needed
if frame_cols ~= N
    frame_matrix = interp2(1:frame_cols, 1:frame_rows, frame_matrix, ...
        linspace(1, frame_cols, N), linspace(1, frame_rows, N)', 'linear');
end

% Calculate the pseudoinverse of the frame matrix
frame_pseudoinverse = pinv(frame_matrix);

% Project the function f onto the frame matrix
coefficients = frame_pseudoinverse * f;

% Reconstruct the function from the frame matrix
f_reconstructed = frame_matrix * coefficients;

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

% Add text annotation to display reconstruction error
annotation_text = sprintf('Reconstruction error: %.5f', reconstruction_error);
dim = [0.2 0.5 0.3 0.3]; % Position of the annotation [x y width height]
annotation('textbox', dim, 'String', annotation_text, 'FitBoxToText', 'on', 'BackgroundColor', 'white');