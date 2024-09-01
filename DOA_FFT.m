function [doa_estimates, angle_vector, power_spectrum, fft_size] = estimate_doa_fft(received_signal, param)
    fft_size = 8;
    nfft = 2^nextpow2(size(received_signal, 1)) * fft_size;  
    d = param.d_E; 

    % Preprocessing the received signal
    received_signal_fft = fftshift(fft(received_signal, nfft, 1), 1);

    % Calculate the power spectrum
    power_spectrum = abs(received_signal_fft).^2;
    power_spectrum = mean(power_spectrum, 2);  

    % Frequency Vector
    fs = 3;  % Sampling frequency
    freq_vector = linspace(-fs/2, fs/2, nfft);

    % Angle Vector
    angle_vector = asind(freq_vector * d);  
    % disp("angle_vector:"+ angle_vector);
    [~, peakIndices] = findpeaks(power_spectrum, 'MinPeakProminence', max(power_spectrum)/100, 'MinPeakDistance', nfft/100);
    doa_estimates = angle_vector(peakIndices);
end
