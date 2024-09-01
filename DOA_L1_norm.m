function [DOAs, residuals] = estimate_DOA_L1(r, param)
    % Input parameters:
    % r - received signal vector
    % G - measurement matrix
    % param - parameter structure containing system parameters

    % Generate dictionary matrix D
    ang_range = linspace(-90, 90, 900);  
    D = exp(1j * 2 * pi * param.d_E * (0:param.M-1).' * sind(ang_range));  % Fourier
    
    % Interference Vector
    % a_AR = param.get_steer(param.theta_AR, param.M);
    b = param.G*param.get_steer(param.theta_AR, param.M);
    % ρ Parameters control the balance between sparsity and reconstruction error
    rho = 0.1; 
    
    cvx_begin quiet
        variable x_hat(size(D, 2), 1) complex
        variable q_hat complex
        minimize( norm(r - param.G * D * x_hat - b * q_hat, 2) + rho * norm(x_hat, 1) )
    cvx_end
    
    % Extracting DOA estimation results
    [~, locs] = findpeaks(abs(x_hat), 'SortStr', 'descend', 'NPeaks', param.K);
    DOAs = ang_range(locs);
    residuals = norm(r - param.G * D * x_hat - b * q_hat, 2);  % 计算重建误差
    
    figure;
    plot(ang_range, 20 * log10(abs(x_hat) / max(abs(x_hat))), 'LineWidth', 2);
    hold on;
    stem(DOAs, zeros(size(DOAs)), 'r', 'LineWidth', 2);
    title('L1 Norm DOA Estimation');
    xlabel('Angle (degrees)');
    ylabel('Normalized Spectrum (dB)');
    grid on;
end

