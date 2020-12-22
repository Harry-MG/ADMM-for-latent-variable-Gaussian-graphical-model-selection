QUIET = 0;
MAX_ITER = 1000;
ABSTOL = 1e-4;
RELTOL = 1e-2;

% Data
C = cov(D);
n = size(C, 1);

% ADMM solver begins
R = zeros(n);
S = zeros(n);
L = zeros(n);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');

for k = 1:MAX_ITER
    
    % R update
    [Q, q] = eig(mu*(sigma-lambda) - S + L);
    es = diag(q);
    xi = 0.5*(-es + sqrt(es.^2 + 4*mu));
    R = Q*diag(xi)*Q';
    
    % S update
    S = shrinkage(mu*lambda - R - L, alpha*mu);
    
    % L update
    Lold = L;
    [U, u] = eig(mu*lambda - S - R);
    evs = diag(u);
    gi = max(evs - mu*beta*ones(n, 1), zeros(n, 1));
    L = U*diag(gi)*U';
    
    % Lagrangian variable update (later - add relaxation)
    lambda = lambda - (R - S + L)/mu;
    
    history.objval(k) = objective(R, S, L, lambda);
    
    history.r_norm(k) = norm(R - S + L, 'fro');
    history.s_norm(k) = norm((l - Lold)/mu, 'fro');
    
    history.eps_pri(k) = n*ABSTOL + RELTOL*max(norm(R, 'fro'), norm(S, 'fro'), norm(L, 'fro'));
    history.eps_dual(k) = n*ABSTOL + RELTOL*norm(lambda, 'fro');
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end

end

function obj = objective(R, S, L, lambda)
    obj = trace(R, sigma) - log(det(R)) + alpha*norm(S, 1) + beta*trace(L) + ...
        - trace(lambda*(R - S + L)) + (norm(R - S + L, 'fro')^2)/2*mu;
end

function shrink = shrinkage(a, kappa)
    shrink = max(0, a - kappa) - max(0, -a - kappa);
end
