# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from learning import a2c_common
from isaacgym.torch_utils import *

import numpy as np
import torch 
from torch import nn
import math
import learning.ultra_agent as ultra_agent


class UltraAgentDistill(ultra_agent.UltraAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.epoch_num_start = 0
        self.expert_loss_coef = config['expert_loss_coef']
        self.entropy_coef = config['entropy_coef']
        self.ev_ma            = 0.0   # running avg explained‑variance
        self.critic_win_streak = 0    # consecutive windows EV ≥ threshold
        self.actor_update_num = 0
        self.vae_dim = config['vae_dim']
        self.aux_loss_coef = config.get('aux_loss_coef', 0.1)
        self.local_goal_pred_coef = config.get('local_goal_pred_coef', 0.1)
        self.disable_critic = config.get('disable_critic', False)
        self.debug_unused_params = config.get('debug_unused_params', False)
        self.vae_noise = torch.zeros(
            self.vec_env.env.task.num_envs,
            self.vae_dim,
            dtype=torch.float,
            device=self.ppo_device,
        )
        self.beta = torch.zeros(
            self.vec_env.env.task.num_envs,
            dtype=torch.float,
            device=self.ppo_device,
        )


    def init_tensors(self):
        super().init_tensors()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['teacher_obs'] = torch.zeros((*batch_shape, 4052), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['expert'] = torch.zeros((*batch_shape, 29), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['vae_noise'] = torch.zeros((*batch_shape, self.vae_dim), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['prior_mu_prev'] = torch.zeros((*batch_shape, self.vae_dim), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['encoder_mu_prev'] = torch.zeros((*batch_shape, self.vae_dim), dtype=torch.float32, device=self.ppo_device)
        self.prior_mu = torch.zeros((self.vec_env.env.task.num_envs, self.vae_dim), dtype=torch.float32, device=self.ppo_device)
        self.encoder_mu = torch.zeros((self.vec_env.env.task.num_envs, self.vae_dim), dtype=torch.float32, device=self.ppo_device)

        self.tensor_list += ['amp_obs', 'rand_action_mask', 'expert', 'expert_mask', 'vae_noise', 'prior_mu_prev', 'encoder_mu_prev', 'teacher_obs']
        return


    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        # Update observation keep probabilities based on current epoch
        if hasattr(self.vec_env.env.task, 'update_obs_keep_probabilities'):
            self.vec_env.env.task.update_obs_keep_probabilities(self.epoch_num)

        # Initialize DAgger beta coefficient
        beta_t = max(1 - max((self.epoch_num - 50) / 500, 0), 0)

        for n in range(self.horizon_length):

            self.obs, self.expert = self.env_reset(self.done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('teacher_obs', n, self.expert['teacher_obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device), self.expert['teacher_obs'].to(self.ppo_device), self.expert['mus'].to(self.ppo_device))
            self.experience_buffer.update_data('vae_noise', n, res_dict['vae_noise'])
            self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]

            if not hasattr(self, 'smooth_reward'):
                self.init_smooth_tensors(infos['smooth_reward_names'])

            self.current_smooth_reward += torch.stack(infos['smooth_reward'] + infos['smooth_reward_buf'], dim=1)       

            self.smooth_reward.update(torch.cat([self.current_smooth_reward[self.done_indices][:, :, :len(infos['smooth_reward_names'])] / self.current_lengths[self.done_indices].unsqueeze(2), self.current_smooth_reward[self.done_indices][:, :, len(infos['smooth_reward_names']):]], dim=2))

            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()
            self.prior_mu = self.prior_mu * not_dones.unsqueeze(1)
            self.encoder_mu = self.encoder_mu * not_dones.unsqueeze(1)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            self.current_smooth_reward = self.current_smooth_reward * (not_dones.unsqueeze(1))
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict


    def get_action_values(self, obs_dict, rand_action_probs, use_experts=0.0, expert=None, teacher_obs=None, experts_mu=None):
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'teacher_obs': teacher_obs.clone(),
            'vae_noise': self.vae_noise.clone(),
            'with_encoder': True,
            'with_vae': True,
            'skip_critic': self.disable_critic
        }


        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        prior_mu = res_dict['rnn_states']['prior_out']['mu']
        encoder_mu = res_dict['rnn_states']['encoder_out']['mu']
        condition = (self.prior_mu.abs().sum(dim=-1) < 1e-5).float().unsqueeze(dim=-1)
        self.prior_mu = condition * prior_mu.clone().detach() + (1 - condition) * self.prior_mu
        self.encoder_mu = condition * encoder_mu.clone().detach() + (1 - condition) * self.encoder_mu

        res_dict['prior_mu_prev'] = self.prior_mu.clone().detach()
        res_dict['encoder_mu_prev'] = self.encoder_mu.clone().detach()

        self.prior_mu = prior_mu.clone().detach()
        self.encoder_mu = encoder_mu.clone().detach()

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        num_envs = self.vec_env.env.task.num_envs

        # Prior rollout: use prior-only actions for some environments
        # Schedule: 0 -> 0.3 over epochs 1500-10000
        progress_prior_rollout = min(max(0, self.epoch_num - 500) / 3000, 1)
        prior_rollout_prob = 0.3 * progress_prior_rollout
        if prior_rollout_prob > 0.01:
            prior_rollout_probs = torch.full((num_envs,), prior_rollout_prob, dtype=torch.float32, device=self.ppo_device)
            use_prior_mask = torch.bernoulli(prior_rollout_probs).bool()
            if use_prior_mask.any():
                # Forward pass with prior only for selected envs
                input_dict_prior = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'rnn_states': self.rnn_states,
                    'vae_noise': self.vae_noise.clone(),
                    'with_encoder': False,  # Prior only
                    # with_vae=True routes through the privileged posterior.
                    'with_vae': False,
                    'skip_critic': True,
                }
                with torch.no_grad():
                    res_dict_prior = self.model(input_dict_prior)
                # Replace actions for selected envs with prior-only actions
                res_dict['actions'][use_prior_mask] = res_dict_prior['mus'][use_prior_mask]
                res_dict['mus'][use_prior_mask] = res_dict_prior['mus'][use_prior_mask]

        expert_action_probs = to_torch([use_experts for _ in range(num_envs)], dtype=torch.float32, device=self.ppo_device)
        expert_action_probs = torch.bernoulli(expert_action_probs)
        det_action_mask = torch.abs(expert_action_probs - 1.0) < 1e-3
        res_dict['actions'][det_action_mask] = expert[det_action_mask]
        res_dict['expert_mask'] = expert_action_probs
        pure_expert = self.beta > 0.95
        res_dict['actions'][pure_expert] = expert[pure_expert]
        res_dict['expert_mask'][pure_expert] = 1.0

        mix_expert = torch.logical_and(self.beta < 0.95, self.beta > 0.90)
        criteria = torch.sum((res_dict['mus'] - experts_mu)**2, dim=-1) > 0.5
        mix_expert = torch.logical_and(mix_expert, criteria)
        res_dict['actions'][mix_expert] = expert[mix_expert]
        res_dict['expert_mask'][mix_expert] = 1.0

        res_dict['vae_noise'] = self.vae_noise.clone()

        return res_dict
    

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        expert = batch_dict['expert']
        expert_mask = batch_dict['expert_mask']
        vae_noise = batch_dict['vae_noise']
        prior_mu_prev = batch_dict['prior_mu_prev']
        encoder_mu_prev = batch_dict['encoder_mu_prev']
        teacher_obs = batch_dict['teacher_obs']
        
        self.dataset.values_dict['expert'] = expert
        self.dataset.values_dict['expert_mask'] = expert_mask
        self.dataset.values_dict['vae_noise'] = vae_noise
        self.dataset.values_dict['prior_mu_prev'] = prior_mu_prev
        self.dataset.values_dict['encoder_mu_prev'] = encoder_mu_prev
        self.dataset.values_dict['teacher_obs'] = teacher_obs

        return


    def _supervise_loss(self, student, teacher):
        e_loss = (student - teacher)**2

        info = {
            'expert_loss': e_loss.sum(dim=-1)
        }
        return info
    
    def kl_loss(self, prior_outs, encoder_outs):
        return 0.5 * (
            prior_outs["logvar"]
            - encoder_outs["logvar"]
            + torch.exp(encoder_outs["logvar"]) / torch.exp(prior_outs["logvar"])
            + encoder_outs["mu"] ** 2 / torch.exp(prior_outs["logvar"])
            - 1
        )
            
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, expert = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, expert #.to(self.ppo_device)
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, expert #.to(self.ppo_device)
        
    def env_reset(self, env_ids=None):
        self.reset_vae_noise(env_ids)
        obs, expert = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs, expert

    def reset_vae_noise(self, env_ids):
        """Reset the VAE noise tensor based on the selected noise type."""
        num_envs = self.vec_env.env.task.num_envs
        vae_latent_dim = self.vae_dim
        if env_ids is None:
            env_ids = torch.arange(num_envs, device=self.ppo_device, dtype=torch.long)
        if type(env_ids) is list:
            env_ids = torch.tensor(env_ids, device=self.ppo_device, dtype=torch.long)
        env_ids = env_ids.to(self.ppo_device)

        noise_type = "normal"
        if noise_type == "normal":
            epsilon = torch.randn(
                env_ids.shape[0], vae_latent_dim, device=self.ppo_device
            )  # sampling epsilon
        elif noise_type == "uniform":
            epsilon = torch.rand(
                env_ids.shape[0], vae_latent_dim, device=self.ppo_device
            )  # sampling epsilon
        elif noise_type == "zeros":
            epsilon = torch.zeros(
                env_ids.shape[0], vae_latent_dim, device=self.ppo_device
            )  # no noise
        else:
            raise NotImplementedError
        self.vae_noise[env_ids] = epsilon
        self.beta[env_ids] = torch.rand(env_ids.shape[0], device=self.device)
        self.prior_mu[env_ids] = torch.zeros(
                env_ids.shape[0], self.prior_mu.shape[1], device=self.ppo_device
            )
        self.encoder_mu[env_ids] = torch.zeros(
                env_ids.shape[0], self.encoder_mu.shape[1], device=self.ppo_device
            )


    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        teacher_obs = input_dict['teacher_obs']
        expert_mus = input_dict['expert']
        vae_noise = input_dict['vae_noise']
        obs_batch = self._preproc_obs(obs_batch)

        expert_mask = (input_dict['expert_mask'] > -1).float()
        expert_sum = torch.sum(expert_mask)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'teacher_obs': teacher_obs.clone(),
            'vae_noise': vae_noise.clone(),
            'with_encoder': True,
            'with_vae': True,
            'skip_critic': self.disable_critic
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            state = res_dict['rnn_states']
            prev_prior_mu = input_dict['prior_mu_prev'].detach()
            prev_encoder_mu = input_dict['encoder_mu_prev'].detach()

            prior_outs, encoder_outs = state['prior_out'], state['encoder_out']

            smooth_loss = 0.0001 * self._loss_mean(((prior_outs['mu'] + encoder_outs['mu'] - prev_prior_mu - prev_encoder_mu)**2).sum(dim=-1))
            
            vae_kld_loss = self.kl_loss(prior_outs, encoder_outs)
            vae_kld_loss = self._loss_mean(torch.sum(vae_kld_loss, dim=-1))

            # Prior-only forward pass: get action using only prior (no encoder/privileged info)
            batch_dict_prior = {
                'is_train': True,
                'prev_actions': actions_batch,
                'obs': obs_batch,
                'vae_noise': vae_noise.clone(),
                'with_encoder': False,  # Only use prior branch
                'with_vae': False,  # Same action path as deployment; latent VAE remains active.
                'skip_critic': True,  # Don't need critic for this
            }
            res_dict_prior = self.model(batch_dict_prior)
            prior_mu = res_dict_prior['mus']
            # Prior sampling loss: match prior-only action to expert action
            prior_action_loss = self._loss_mean(((prior_mu - expert_mus)**2).sum(dim=-1))

            # Smooth cosine schedule for KLD coefficient: 0.001 -> 0.2 over epochs 500-2000
            progress = min(max(0, self.epoch_num - 500) / 3000, 1)
            kld_coeff = 0.001 + (0.2 - 0.001) * (1 - torch.cos(torch.tensor(progress * 3.14159265))) / 2

            # Smooth cosine schedule for prior action loss coefficient: 0 -> 0.3 over epochs 500-2000
            progress_prior = min(max(0, self.epoch_num - 500) / 3000, 1)
            prior_action_coeff = 0.6 * (1 - torch.cos(torch.tensor(progress_prior * 3.14159265))) / 2

            # Smooth cosine schedule for smooth coefficient: 0.0001 -> 0.001 over epochs 500-1500
            progress_smooth = min(max(0, self.epoch_num - 500) / 3000, 1)
            smooth_coeff = 0.0001 + (0.001 - 0.0001) * (1 - torch.cos(torch.tensor(progress_smooth * 3.14159265))) / 2
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            c_loss = self._loss_mean(c_loss)
            a_loss = self._loss_mean(a_loss) 
            entropy = self._loss_mean(entropy) 
            b_loss = self._loss_mean(b_loss) 
            a_clip_frac = self._loss_mean(a_clipped) 

            e_info = self._supervise_loss(mu, expert_mus)
            e_loss = e_info['expert_loss']
            e_loss = self._loss_mean(e_loss)
            aux_loss = self._auxiliary_loss(state.get('aux_pred'), obs_batch)
            local_goal_loss = self._local_goal_pred_loss(state.get('local_goal_pred'), obs_batch)
            loss =  0 * (a_loss + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss) + \
                    self.expert_loss_coef * e_loss + vae_kld_loss * kld_coeff + smooth_loss * smooth_coeff + \
                    aux_loss * self.aux_loss_coef + local_goal_loss * self.local_goal_pred_coef + \
                    prior_action_loss * prior_action_coeff
            
            # a_info['actor_loss'] = a_loss
            # a_info['actor_clip_frac'] = a_clip_frac
            # c_info['critic_loss'] = c_loss
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.debug_unused_params:
            if (not self.multi_gpu) or torch.distributed.get_rank() == 0:
                unused = [
                    name
                    for name, param in self.model.named_parameters()
                    if param.requires_grad and param.grad is None
                ]
                if unused:
                    print("[UltraAgentDistill] Unused params:", unused)
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
            self.train_result = {
                'entropy': entropy,
                'kl': kl_dist,
                'last_lr': self.last_lr,
                'lr_mul': lr_mul,
                'kl_loss': vae_kld_loss,
                'smooth_loss': smooth_loss,
                'aux_loss': aux_loss,
                'local_goal_loss': local_goal_loss,
                'prior_action_loss': prior_action_loss,
            }
        # self.train_result.update(a_info)
        # self.train_result.update(c_info)
        self.train_result.update(e_info)
        return

    def _eval_critic(self, obs_dict):
        if self.disable_critic:
            return torch.zeros(
                obs_dict['obs'].shape[0],
                self.value_size,
                device=self.ppo_device,
                dtype=torch.float32,
            )
        return super()._eval_critic(obs_dict)
    
    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        # self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        # self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        self.writer.add_scalar('losses/e_loss', torch_ext.mean_list(train_info['expert_loss']).item(), frame)
        
        # self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/kl_loss', torch_ext.mean_list(train_info['kl_loss']).item(), frame)
        self.writer.add_scalar('losses/s_loss', torch_ext.mean_list(train_info['smooth_loss']).item(), frame)
        self.writer.add_scalar('losses/aux_loss', torch_ext.mean_list(train_info['aux_loss']).item(), frame)
        self.writer.add_scalar('losses/local_goal_loss', torch_ext.mean_list(train_info['local_goal_loss']).item(), frame)
        self.writer.add_scalar('losses/prior_action_loss', torch_ext.mean_list(train_info['prior_action_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        # self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)

        self.writer.add_scalar('usage/cpu', self.get_cpu_usage(), frame)
        self.writer.add_scalar('usage/gpu', self.get_gpu_usage(), frame)
        self.writer.add_scalar('usage/cpu_memory', self.get_cpu_memory_usage(), frame)
        self.writer.add_scalar('usage/gpu_memory', self.get_gpu_memory_usage(), frame)

        return

    def _local_goal_pred_loss(self, pred, obs_batch):
        if pred is None:
            return torch.zeros(1, device=self.ppo_device)

        modal_cfg = getattr(self.model.a2c_network, "modal_cfg", None)
        if modal_cfg is None:
            return torch.zeros(1, device=self.ppo_device)

        goal_dim = modal_cfg.get("goal_dim", 0)
        command_dim = modal_cfg.get("command_dim", 0)
        local_goal_dim = modal_cfg.get("local_goal_dim", 0)
        task_dim = modal_cfg.get("task_obs_dim", 0)
        mask_dim = modal_cfg.get("mask_dim", 0)

        if local_goal_dim <= 0:
            return torch.zeros(1, device=self.ppo_device)

        obs = obs_batch
        if obs.shape[-1] < mask_dim:
            return torch.zeros(1, device=obs.device)

        mask_slice = obs[..., -mask_dim:]
        data_len = obs.shape[-1] - mask_dim
        body_dim = max(0, data_len - (goal_dim + command_dim + local_goal_dim + task_dim))
        local_goal_start = goal_dim + command_dim
        local_goal = obs[..., local_goal_start : local_goal_start + local_goal_dim]

        mask_local = mask_slice[..., goal_dim : goal_dim + local_goal_dim]
        weight = 1.0 - mask_local
        diff = (pred - local_goal) ** 2
        diff = diff * weight
        return self._loss_mean(diff.mean(dim=-1))

    def _auxiliary_loss(self, aux_pred, obs_batch):
        if aux_pred is None:
            return torch.zeros(1, device=self.ppo_device)

        modal_cfg = getattr(self.model.a2c_network, "modal_cfg", None)
        if modal_cfg is None:
            return torch.zeros(1, device=self.ppo_device)

        goal_dim = modal_cfg.get("goal_dim", 0)
        command_dim = modal_cfg.get("command_dim", 0)
        local_goal_dim = modal_cfg.get("local_goal_dim", 0)
        obj_trans_dim = modal_cfg.get("obj_trans_dim", 0)
        obj_rot_dim = modal_cfg.get("obj_rot_dim", 0)
        obj_pos_dim = modal_cfg.get("obj_pos_dim", 0)
        task_dim = modal_cfg.get("task_obs_dim", 0)
        raw_point_dim = modal_cfg.get("raw_point_obs_dim", 0)
        mask_dim = modal_cfg.get("mask_dim", 0)

        obs = obs_batch
        if obs.shape[-1] < mask_dim:
            return torch.zeros(1, device=obs.device)

        mask_slice = obs[..., -mask_dim:]
        data_len = obs.shape[-1] - mask_dim
        body_dim = max(0, data_len - (goal_dim + command_dim + local_goal_dim + task_dim))

        task_start = goal_dim + command_dim + local_goal_dim + body_dim
        task_slice = obs[..., task_start : task_start + task_dim]

        global_goal = obs[..., :goal_dim] if goal_dim > 0 else torch.zeros_like(aux_pred[..., :0])
        command = obs[..., goal_dim : goal_dim + command_dim] if command_dim > 0 else torch.zeros_like(aux_pred[..., :0])
        obj_trans = task_slice[..., :obj_trans_dim] if obj_trans_dim > 0 else torch.zeros_like(aux_pred[..., :0])
        obj_rot = task_slice[..., obj_trans_dim : obj_trans_dim + obj_rot_dim] if obj_rot_dim > 0 else torch.zeros_like(aux_pred[..., :0])

        target = torch.cat([command, global_goal, obj_trans, obj_rot], dim=-1)

        idx = 0
        mask_goal = mask_slice[..., idx : idx + goal_dim]; idx += goal_dim
        idx += local_goal_dim
        mask_trans = mask_slice[..., idx : idx + obj_trans_dim]; idx += obj_trans_dim
        mask_rot = mask_slice[..., idx : idx + obj_rot_dim]; idx += obj_rot_dim
        idx += obj_pos_dim
        idx += raw_point_dim
        mask_command = mask_slice[..., idx : idx + command_dim]

        mask_targets = []
        if command_dim > 0:
            mask_targets.append(mask_command)
        if goal_dim > 0:
            mask_targets.append(mask_goal)
        if obj_trans_dim > 0:
            mask_targets.append(mask_trans)
        if obj_rot_dim > 0:
            mask_targets.append(mask_rot)
        if len(mask_targets) == 0:
            return torch.zeros(1, device=obs.device)
        target_mask = torch.cat(mask_targets, dim=-1)

        min_len = min(target.shape[-1], target_mask.shape[-1], aux_pred.shape[-1])
        if min_len == 0:
            return torch.zeros(1, device=obs.device)
        target = target[..., :min_len]
        target_mask = target_mask[..., :min_len]
        aux_pred = aux_pred[..., :min_len]

        diff = (aux_pred - target) ** 2
        weighted = diff * target_mask
        denom = target_mask.sum(dim=-1, keepdim=True) + 1e-6
        per_sample = weighted.sum(dim=-1) / denom.squeeze(-1)
        return self._loss_mean(per_sample)
