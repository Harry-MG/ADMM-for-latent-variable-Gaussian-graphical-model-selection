randn('seed', 0);
rand('seed', 0);

p = 1000;
ph = 10;

%parameter tuning
mu = 10;
alpha = 1;
beta = 1;

t_inc = 2;
t_dec = 2;
var_par = 10;

n = p+ph;   % number of features
N = 5*p;  % number of samples

% generate a sparse positive definite inverse covariance matrix
Kinv      = diag(abs(ones(n,1)));
idx       = randsample(n^2, 0.001*n^2);
Kinv(idx) = ones(numel(idx), 1);
Kinv = Kinv + Kinv';   % make symmetric
if min(eig(Kinv)) < 0  % make positive definite
    Kinv = Kinv + 1.1*abs(min(eig(Kinv)))*eye(n);
end
leng = inv(Kinv);

%Ground truth S matrix
S = K(1:p, 1:p);

%Groud truth L matrix
L = K(1:p, p+1:n)*inv(K(p+1:n, p+1:n))K(p+1:n, 1:p);

% generate Gaussian samples
D = mvnrnd(zeros(1,p), leng, N);

%run the solver

[X, history] = latentcovsel(D, mu, alpha, beta, t_inc, t_dec, var_par);

%Reporting

leng = length(history.objval);
X_admm = X;

h = figure;
plot(1:leng, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(R^k) + g(S^k) + h(L^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);
semilogy(1:leng, max(1e-8, history.r_norm), 'k', ...
    1:leng, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:leng, max(1e-8, history.s_norm), 'k', ...
    1:leng, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');