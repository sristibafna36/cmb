%%  Modelling Competition 
%   PSY-3102-1 
%   Model: Reinforcement Learning (Unbiased) + RW Learning Rule
%   Function 

function [choices, rewards, choice_probabilities, Q1, Q2]  = RL_unbiased_func(param,T,u,Q_init,rescale,states)
%%  Cleanup 

clc; 
close all; 

%% 

beta = param(1);
alpha_opt = param(2);   % positivity bias learning rate
alpha_pes = param(3);   % negativity bias learning rate 


%%  Setting output variables - Positivity Bias 

Q = zeros(2,2) + Q_init;    % creating placeholder vectors of zeros to generate initial Q values 

t = 0;

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
    
    Q(states(t), choices(t)) = Q(states(t), choices(t)) + alpha_opt*prediction_error*(prediction_error>0) + alpha_pes*prediction_error*(prediction_error<0); 

    Q1(t) = Q(states(t),1); % Q val for option 1 given a state on trial t 
    Q2(t) = Q(states(t),2); % Q val for option 2 give a state on trial t

end % end the trial loop

end 



%Q(t+1,:) = Q(t,:); % first copy the current value estimates over to the next row of Q
    %if prediction_error >= 0 
     %   Q(t,choices(t,1)) = Q(t,choices(t,1)) + alpha_opt(n,2)*prediction_error; % Optimistic alpha: update the value estimate of the chosen option to make next choice (t+1)
    %else
     %   Q(t,choices(t,1)) = Q(t,choices(t,1)) + alpha_pes(n,3)*prediction_error; % Pessimistic alpha: update the value estimate of the chosen option to make next choice (t+1)                                                                                                                       
        




