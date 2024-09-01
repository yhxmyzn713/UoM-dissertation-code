clear;
% Initialization parameters
param.vec = @(MAT) MAT(:);
param.vecH = @(MAT) MAT(:).';
rng(111);
param.theta_RS = 0;
param.d_E = 0.5;  
param.K = 3;  
param.d_AT = 20;
param.d_TR = 30;
param.d_RS = 3;
param.d_AR = 5;
param.N = 16;  % the number of measurements
param.get_steer = @(theta, L) exp(1j*2*pi*[0:1:L-1].'*param.d_E*param.vecH(sind(theta+param.theta_RS)));

z = 1/(param.d_AT*param.d_TR*param.d_RS) * exp(1j*rand(param.K,1));
q = 1/(param.d_AR*param.d_RS) * exp(1j*rand(1));
param.theta_TR = [-25, 15, 30].' + rand(param.K,1);

SNR_dB = 20;
param.Ms = 8:9:62;  % the range of the number of RIS elements
rmse_fft = zeros(1, length(param.Ms));
rmse_proposed = zeros(1, length(param.Ms));
rmse_crlb = zeros(1, length(param.Ms));
rmse_l1norm = zeros(1, length(param.Ms));

for i = 1:length(param.Ms)
    param.M = param.Ms(i);
    disp("param.Ms："+param.Ms(i));
    param.theta_AR = -10 + rand(1);

    a_tmp = param.get_steer(param.theta_AR, param.M);
    A_tmp = a_tmp * a_tmp';
    cvx_begin sdp quiet
        variable G_tilde(param.M, param.M) hermitian
        minimize(trace(A_tmp * G_tilde))
        subject to
            G_tilde >= 0
            for idx_m = 1:param.M
                G_tilde(idx_m, idx_m) == 1;
            end
    cvx_end

    [U_tilde, D_tilde] = eig(G_tilde);
    D_tilde = diag(sqrt(real(diag(D_tilde))));
    N_test = 1e4;
    g_tilde = sqrt(1/2) * (randn(param.M, N_test) + 1j * randn(param.M, N_test));
    G_tmp = exp(1j * angle(U_tilde * D_tilde * g_tilde));
    G = G_tmp';
    param.G = G(1:param.N, :);

    param.dic_ang = [-45:1:45].'; % for grid estimation
    param.dic_mat = param.get_steer(param.dic_ang, param.M);

    param.cont_ang = [-45:0.001:45].'; % for off-grid estimation
    param.cont_dic  = param.get_steer(param.cont_ang, param.M);

    r_z = param.G * param.get_steer(param.theta_TR, param.M) * z;
    r_q = param.G * param.get_steer(param.theta_AR, param.M) * q;
    
    noise_pow = norm(r_z)^2 / 10^(SNR_dB/10);

    w_temp = sqrt(1/2) * (randn(size(r_z)) + 1j * randn(size(r_z)));
    noise_var = noise_pow / norm(w_temp)^2;
    w = sqrt(noise_var) * w_temp;
    % Adding noise to the received signal
    r = r_z + r_q + w;  

    [fft_doa_estimates, fft_angle_vector, fft_power_spectrum,fft_size] = estimate_doa_fft(r, param);
    est_spectrum = generate_est_spectrum(fft_angle_vector, fft_power_spectrum);
    rmse_fft(i) = calculate_rmse(est_spectrum, param.theta_TR);

    [est_x, est_spectrum] = apply_proposed_method(r, param, noise_var);
    rmse_proposed(i) = calculate_rmse(est_spectrum, param.theta_TR);

    rmse_crlb(i) = calculate_crlb(z, r_z, param, noise_pow);

    %-----------------------L1_norm-------------------------------------------------%
    ang_range = linspace(-90, 90, 180);  
    D = exp(1j * 2 * pi * param.d_E * (0:param.M-1).' * sind(ang_range));  % Fourier 
    a_AR = param.get_steer(param.theta_AR, param.M);
    % ρ Parameters control the balance between sparsity and reconstruction error
    rho = 1; 
    
    cvx_begin quiet
        variable x_hat(size(D, 2), 1) complex
        variable q_hat complex
        % 最小化目标函数    
        minimize( norm(r - param.G * D * x_hat - param.G * a_AR * q_hat, 2) + rho * norm(x_hat, 1) )
    cvx_end

    est_spectrum = [ang_range(:), abs(x_hat(:))];
    rmse_l1norm(i) = calculate_rmse(est_spectrum, param.theta_TR);
    %---------------------------------------------------------------------------------%

end

plot_results(param.Ms, rmse_fft, rmse_proposed, rmse_l1norm, rmse_crlb);

%% function
function [est_spectrum] = generate_est_spectrum(fft_angle_vector, fft_power_spectrum)
    est_spectrum = [fft_angle_vector(:), fft_power_spectrum(:)];
end

function rmse = calculate_rmse(est_spectrum, theta_TR)
    [pks, pks_idx] = findpeaks(est_spectrum(:, 2));
    est_ang = zeros(length(theta_TR), 1);
    if ~isempty(pks_idx)
        for idx = 1:length(theta_TR)
            [~, min_idx] = min(abs(theta_TR(idx) - est_spectrum(pks_idx, 1)));
            est_ang(idx) = est_spectrum(pks_idx(min_idx), 1);
        end
    end
    rmse = sqrt(norm(est_ang - theta_TR)^2 / length(theta_TR));
end


% proposed method
function [est_x, est_spectrum] = apply_proposed_method(r, param, noise_var)
    b = param.G * param.get_steer(param.theta_AR, param.M);
    rho = sqrt(log(param.M) * param.M) * sqrt(noise_var);
    cvx_begin sdp quiet
        variable est_xi(param.M, 1) complex
        variable u(param.M, 1) complex
        variable est_eta complex
        variable Z(param.M, param.M) hermitian toeplitz
        variable nu(1, 1)
        minimize(quad_form(r - param.G * est_xi - est_eta * b, eye(length(r))) + rho / 2 * (nu + 1/param.M * trace(Z)))
        subject to
            [Z, est_xi; est_xi', nu] >= 0
            Z(:, 1) == u
    cvx_end

    est_x = MUSIConesnapshot(est_xi, param);
    est_spectrum = [param.cont_ang, zeros(length(param.cont_ang), 1)];
    est_spectrum(:, 2) = abs(est_x);
end

function rmse_crlb = calculate_crlb(z, r_z, param,noise_pow)
    B = zeros(param.M, param.K);
    for idx = 1:param.K
        B(:, idx) = 1j * 2 * pi * param.d_E * z(idx) * cosd(param.theta_TR(idx) + param.theta_RS) * (param.get_steer(param.theta_TR(idx), param.M) .* [0:param.M-1].');
    end

    F = length(r_z)/noise_pow*[2*real(B'*param.G'*param.G*B), B'*param.G'*param.G*param.get_steer(param.theta_TR, param.M), B'*param.G'*param.G*param.get_steer(param.theta_AR, param.M);
	param.get_steer(param.theta_TR, param.M)'*param.G'*param.G*B, param.get_steer(param.theta_TR, param.M)'*param.G'*param.G*param.get_steer(param.theta_TR, param.M), param.get_steer(param.theta_TR, param.M)'*param.G'*param.G*param.get_steer(param.theta_AR, param.M);
	param.get_steer(param.theta_AR, param.M)'*param.G'*param.G*B, param.get_steer(param.theta_AR, param.M)'*param.G'*param.G*param.get_steer(param.theta_TR, param.M), param.get_steer(param.theta_AR, param.M)'*param.G'*param.G*param.get_steer(param.theta_AR, param.M)];
    crlb_all = abs(diag(inv(F)));
    crlb = rad2deg(sqrt(sum(crlb_all(1:param.K))/param.K));
    rmse_crlb = rad2deg(sqrt(sum(crlb_all(1:param.K)) / param.K));
end

function plot_results(Ms, rmse_fft, rmse_proposed, rmse_l1norm, rmse_crlb)
    figure;
    semilogy(Ms, rmse_fft, '-bs', 'DisplayName', 'FFT','LineWidth',2,'MarkerFaceColor', 'b', 'MarkerSize', 6);
    hold on;
    semilogy(Ms, rmse_proposed, '-ro', 'DisplayName', 'Proposed Method','LineWidth',2,'MarkerFaceColor', 'r', 'MarkerSize', 6);
    semilogy(Ms, rmse_l1norm, '-+', 'DisplayName', 'L1-norm','LineWidth',2,'MarkerFaceColor', 'y', 'MarkerSize', 6);
    semilogy(Ms, rmse_crlb, '-g', 'DisplayName', 'CRLB','LineWidth',2,'MarkerFaceColor', 'g', 'MarkerSize', 6);
    xlabel('The number of RIS elements');
    ylabel('RMSE (deg)');
    legend show;
    grid on;
    %title('Fig.8.The estimation performance with different numbers of RIS elements');
    set(gca, 'YScale', 'log');
    xlim([8 64]);
    ylim([1e-2 10]); 
end
