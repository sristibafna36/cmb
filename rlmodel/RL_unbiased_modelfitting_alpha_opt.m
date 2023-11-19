%%  Modelling Competition 
%   PSY-3102-1
%   Model: Reinforcement Learning (Unbiased) + RW Learning Rule
%   Model Fitting: Optimistic Alpha (Positivity Bias) 

%%  Cleanup 
clc;
clear; 
close all; 

%%  Loading Data 
test_data = load_test_data;
testdata.choice
T = 100; 

%%  Setting Simulation 

nIter = 100;    % Running 100 model fits 

%%  Looping Through Trials 
%   2 for loops, pos, neg alpha 

for i = 1:nIter

    startpt = [(0.3)+(rand*0.4), (2)+(rand*0.3)];  % setting random starting points for fmincon function
    lb = [0, 0];  % setting lower bound for the optimistic alpha and beta
    ub = [1, 1];   % setting upper bound for the optimistic alpha and beta 

    [res(i,:), lik(i), ~,~,~,~,~,] = ... 
        fmincon(@(x) RL_unbiased_nll(choicesRe2, rewards, T, x(1), x(2)), startpt,[],[],[],[],lb,ub); 
end 

