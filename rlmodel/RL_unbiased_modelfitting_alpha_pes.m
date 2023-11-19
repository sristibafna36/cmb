%%  Modelling Competition 
%   PSY-3102-1
%   Model: Reinforcement Learning (Unbiased) + RW Learning Rule
%   Model Fitting: Pessimistic Alpha (Negativity Bias) 

%%  Cleanup 
clc;
clear; 
close all; 

%%  Loading Data 
[test_data_patients] = load_test_data_patients(100);
test_data_controls.trial
test_data_controls.choice
test_data_controls.reward
T = 100; 

%%  Setting Simulation 

nIter = 100;    % Running 100 model fits 

%%  Looping Through Trials 

for i = 1:nIter

    startpt = [(0.3)+(rand*0.4), (2)+(rand*0.3)];  % setting random starting points for fmincon function
    lb = [0, 0];  % setting lower bound for the pessimistic alpha and beta
    ub = [1, 1];   % setting upper bound for the pessimistic alpha and beta 

    [res(i,:), lik(i), ~,~,~,~,~,] = ... 
        fmincon(@(x) RL_unbiased_nll(choicesRe3, rewards, T, x(1), x(2)), startpt,[],[],[],[],lb,ub); 
end 

