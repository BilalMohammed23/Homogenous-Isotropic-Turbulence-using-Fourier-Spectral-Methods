clear
close all
clc

Nx = 32;

L = 2*pi;
h = L/Nx;
j = (0:Nx-1);
x = j*h;

l = [0 : (Nx/2)-1 , -Nx/2 : -1];
m = l;
n = l;

[ll, mm, nn] = ndgrid(l,m,n);

kx = (2*pi/L)*ll;
ky = (2*pi/L)*mm;
kz = (2*pi/L)*nn; 

%%
%initial condition for isotropic turbulence
u0 = 1; %rms vel prescribed
k0 = 5;  %prescribed k0

Re = 40;

lamda_taylor_mcrscale = 2/k0;

diff_gamma = u0 * lamda_taylor_mcrscale/Re;

k_mag = sqrt(kx.^2 + ky.^2 + kz.^2);

[uhatk0, vhatk0, whatk0, u, v, w , COM, q_sq_rms , COM_zero_check , max_rel_err, E] = uvw_hatk_ic_correct_function(Nx, l, m, n, ll, mm, nn, kx, ky, kz, u0, k0, k_mag);

q2_0 = mean(u(:).^2 + v(:).^2 + w(:).^2);

uhatk_old = uhatk0;
vhatk_old = vhatk0;
whatk_old = whatk0;

%%
%

alpha_RK4 = 2.79;

uhatk_np_1_RK4 = zeros(Nx,Nx,Nx);
vhatk_np_1_RK4 = zeros(Nx,Nx,Nx);
whatk_np_1_RK4 = zeros(Nx,Nx,Nx);

max_u = max(abs(u(:)));
max_v = max(abs(v(:)));
max_w = max(abs(w(:)));
max_vel = max([max_u max_v max_w]);

T = 10;
t = 0;

count = 0;

nmax = 10000;           
time_hist  = zeros(nmax,1);
q2_num_norm_hist = zeros(nmax,1);
S_hist = zeros(nmax,1);

%%
while t < T
    delta_t_max = dt(max_vel,alpha_RK4, diff_gamma,h);
    
    if t + delta_t_max > T
        delta_t_max = T - t;
    end

    [uhatk_np_1_RK4 , vhatk_np_1_RK4 , whatk_np_1_RK4] = RK4_PsuedoSpectral_Isotropic(Nx, uhatk_old, vhatk_old, whatk_old, kx, ky, kz, k_mag,delta_t_max,diff_gamma);

    uhatk_old = uhatk_np_1_RK4;
    vhatk_old = vhatk_np_1_RK4;
    whatk_old = whatk_np_1_RK4;

    u_num = real(ifftn(uhatk_old));
    v_num = real(ifftn(vhatk_old));
    w_num = real(ifftn(whatk_old));
    
    %% Pressure calculation
    P_hat = pressure_poisson(Nx, uhatk_old, vhatk_old, whatk_old, kx, ky, kz, k_mag);
    p_num = real(ifftn(P_hat));

    %%

    max_u = max(abs(u_num(:)));
    max_v = max(abs(v_num(:)));
    max_w = max(abs(w_num(:)));
    max_vel = max([max_u max_v max_w]);

    t = t + delta_t_max;
    count = count + 1;

    COM = (kx .* uhatk_old) + (ky .* vhatk_old) + (kz .* whatk_old);
    max_COM = max(abs(COM(:)));

    q2_num = mean(u_num(:).^2 + v_num(:).^2 + w_num(:).^2);
    q2_num_norm   = q2_num   / q2_0;
    
    %velocity deriv skewness
    du_dx = real(ifftn(1i .* kx .* uhatk_old));
    dv_dy = real(ifftn(1i .* ky .* vhatk_old));
    dw_dz = real(ifftn(1i .* kz .* whatk_old));

    m2_1 = mean(du_dx(:).^2);  
    m3_1 = mean(du_dx(:).^3);

    m2_2 = mean(dv_dy(:).^2);  
    m3_2 = mean(dv_dy(:).^3);

    m2_3 = mean(dw_dz(:).^2);  
    m3_3 = mean(dw_dz(:).^3);

    S1 = m3_1 / (m2_1^(3/2) + eps);
    S2 = m3_2 / (m2_2^(3/2) + eps);
    S3 = m3_3 / (m2_3^(3/2) + eps);

    S  = (S1 + S2 + S3)/3;
    
    T_tau = 2/(u0 * k0);
    time_hist(count)  = t *(1 / T_tau);
    q2_num_norm_hist(count)   = q2_num_norm;
    S_hist(count)           = S;

    fprintf('%4d: t = %.4f, dt = %.3e, max_vel = %.3e, max div = %.3e, Skewness = %.3e ,q2_normalized = %.3e\n', count, t, delta_t_max, max_vel ,max_COM , S, q2_num_norm);

end

%% q_squared vs time plot

time_hist = time_hist(1:count);
q2_num_norm_hist   = q2_num_norm_hist(1:count);
S_hist = S_hist(1:count);

u_if = u_num;
v_if = v_num;
w_if = w_num;

figure(1);
plot(time_hist, q2_num_norm_hist, 'LineWidth', 2);
xlabel('Time(normalized)');
ylabel('q^2(t) / q_0^2');
title('Decay of normalized energy q^2 vs time');
grid on;
legend(sprintf('Re = %d' ,Re));
hold on

%% 2D visualization velocity

iz = round(Nx/2); %z-plane (middle)

U_slice = sqrt(u_if(:,:,iz).^2 + v_if(:,:,iz).^2 + w_if(:,:,iz).^2);

x_tile = linspace(0, L, Nx); 
y_tile = linspace(0, L,   Nx);  

figure(2);
imagesc(x_tile, y_tile, U_slice);
set(gca, 'YDir', 'normal');
axis equal tight
colormap(jet);
colorbar;
xlabel('x'); ylabel('y');
title(sprintf('Velocity magnitude in z-mid plane, Re = %d', Re));
hold on;

%% Velocity skewness

figure(3)
plot(time_hist, S_hist, 'LineWidth', 2);
xlabel('Time(normalized)');
ylabel('Skewness');
title('Velocity derivative Skewness plot');
grid on;
legend(sprintf('Re = %d' ,Re));
hold on

%% 2D visualization Pressure

iz = round(Nx/2); %z-plane (middle)

P_slice = sqrt(p_num(:,:,iz).^2);

x_tile = linspace(0, L, Nx); 
y_tile = linspace(0, L,   Nx);  

figure(4);
imagesc(x_tile, y_tile, P_slice);
set(gca, 'YDir', 'normal');
axis equal tight
colormap(jet);
colorbar;
xlabel('x'); ylabel('y');
title(sprintf('Pressure contour, Re = %d', Re));

%%
