%%  Modelling Competition 
%   PSY-3102-1
%   Model: Reinforcement Learning (Unbiased) + RW Learning Rule
%   Negative Log Likelihood Function - Pessimistic Alpha (Negativity Bias) 

function [nll_alpha_pes] = RL_unbiased_nll_alpha_pes(choices, rewards,T, alpha_pes, beta) 
%%  Cleanup 

clc; 
close all;
%%  Setting Variables 

choice_probabilities= NaN(T,2); % a placeholder vector of NaNs to record choice probabilities on every trial
logptrial_alpha_pes = NaN(T,1); % placeholder vector of NaNs to record log likelihood per trial
Q_init = [0.5 0.5]; % Initial estimate of the value of each option (i.e. on trial 0). 
Q = NaN(T,2); % a placeholder matrix of NaNs to record value estimates
Q(1,:) = Q_init;

%% Calculating log likelihood per trial for "optimistic alpha" representing positivity bias 

for i = 1:T % loop through trials

    t = t+1; 

    % Computing the choice probabilities based on current value estimates
    % based on Softmax rule 
    choice_probabilities(t) = 1/ (1 + exp(((Q(states(t),1) - Q(states(t),2))/ beta)));    
                                                                            
    % Make choice between 2 options 
    choices(t) = (choice_probabilities(t)> rand) + 1;
    
    % Deliver the reward based on reward probabilities
    rewards(t) = (rand>u(states(t), choices(t))) - 0.5*rescale;


    % Updating the value estimates based on the Rescorla-Wagner rule
    prediction_error = (rewards(t) - Q(states(t),choices(t)));  % the prediction error = the expected reward and the actual reward
    
%if prediction_error > 0 
    Q(states(t), choices(t)) = Q(states(t), choices(t)) + alpha_pes*prediction_error*(prediction_error>0);
%else
  %  Q(states(t), choices(t)) = Q(states(t), choices(t)) + alpha_pes*prediction_error*(prediction_error<0); 

    Q1(t) = Q(states(t),1); % Q val for option 1 given a state on trial t 
    Q2(t) = Q(states(t),2); % Q val for option 2 give a state on trial t

                                                                                                                     
logptrial_alpha_pes(t) = log(choice_probabilities(choices(t)));  % calculating log likelihood for each trial 

end % end the trial loop

loglikesum_alpha_pes = sum(logptrial_alpha_pes);    % summing log likelihoods across trials

nll_alpha_pes = (-1)*(loglikesum_alpha_pes);        % calculating negative log likelihood

end 