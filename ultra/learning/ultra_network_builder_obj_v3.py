import copy
import numpy as np
import torch
import torch.nn as nn

from learning import network_builder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer tokens."""

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if self.pe.shape[1] < x.shape[1]:
            raise ValueError(
                f"PositionalEncoding max_len={self.pe.shape[1]} is smaller than sequence {x.shape[1]}"
            )
        return x + self.pe[:, : x.shape[1]]


class SimplePointEncoder(nn.Module):
    """Permutation-invariant encoder that keeps coarse object geometry."""

    def __init__(self, point_dim=3, hidden_dim=128, output_dim=128):
        super().__init__()
        self.point_dim = point_dim
        self.output_dim = output_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        global_dim = hidden_dim * 2 + 3 + 3 + 9
        global_hidden = max(hidden_dim, output_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, output_dim),
            nn.ReLU(),
        )

    def forward(self, points):
        """
        Args:
            points: [batch, num_points * point_dim]
        """
        batch_size = points.shape[0]
        num_points = max(1, points.shape[1] // max(1, self.point_dim))
        if num_points == 0:
            return torch.zeros(
                batch_size, self.output_dim, device=points.device, dtype=points.dtype
            )

        points = points.reshape(batch_size, num_points, self.point_dim)
        centroid = points.mean(dim=1)
        rel_points = points - centroid.unsqueeze(1)

        point_features = self.point_mlp(rel_points)
        global_max = torch.max(point_features, dim=1).values
        global_mean = torch.mean(point_features, dim=1)

        axis_extents = torch.max(points, dim=1).values - torch.min(points, dim=1).values
        covariance = torch.bmm(rel_points.transpose(1, 2), rel_points) / float(num_points)
        covariance = covariance.reshape(batch_size, -1)

        global_features = torch.cat(
            [centroid, axis_extents, covariance, global_max, global_mean], dim=1
        )
        return self.global_mlp(global_features)


class UltraBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        @staticmethod
        def _parse_int_list(raw, default):
            if raw is None:
                return list(default)
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.replace(":", ",").split(",")]
                vals = [int(p) for p in parts if p]
            elif isinstance(raw, (list, tuple)):
                vals = [int(v) for v in raw]
            else:
                vals = [int(raw)]
            vals = sorted({v for v in vals if v > 0})
            return vals or list(default)

        def __init__(self, params, point_cloud_cfg=None, **kwargs):
            self.point_cloud_cfg = point_cloud_cfg or {}
            self._input_shape = kwargs.get("input_shape")
            params_local = copy.deepcopy(params)
            if params_local.get("disable_double_critic", True):
                params_local["separate"] = False
            super().__init__(params_local, **kwargs)

            self.obs_dim = int(np.prod(self._input_shape))
            self.modal_cfg = params_local.get("modal_config", {})
            self.vae_dim = params_local.get("vae_dim", self.modal_cfg.get("vae_dim", 64))
            self.debug_modal_layout = self.modal_cfg.get("debug_modal_layout", False)
            self.mask_routing = self.modal_cfg.get("mask_routing", True)
            self.critic_separate_backbone = self.modal_cfg.get("critic_separate_backbone", False)
            self.critic_mlp_hidden = self.modal_cfg.get("critic_mlp_hidden", 512)
            self.critic_mlp_layers = self.modal_cfg.get("critic_mlp_layers", 2)

            self.token_dim = self.modal_cfg.get("token_dim", 192)
            self.modal_hidden = self.modal_cfg.get("fusion_hidden", 384)
            self.transformer_heads = self.modal_cfg.get("transformer_heads", 2)
            self.transformer_layers = self.modal_cfg.get("transformer_layers", 2)
            self.transformer_ffn = self.modal_cfg.get(
                "transformer_ffn_dim", self.token_dim * 4
            )
            self.transformer_dropout = self.modal_cfg.get("transformer_dropout", 0.0)
            self.film_scale = self.modal_cfg.get("film_scale", 0.1)
            self.teacher_obs_dim = self.modal_cfg.get("teacher_obs_dim", 4052)
            self.latent_prior_mode = str(
                self.modal_cfg.get("latent_prior_mode", "gaussian_residual")
            ).lower()
            valid_prior_modes = {
                "gaussian_residual",
                "fixed_uniform_sphere",
                "learned_sphere_prior",
            }
            if self.latent_prior_mode not in valid_prior_modes:
                raise ValueError(
                    "modal_config.latent_prior_mode must be one of "
                    f"{sorted(valid_prior_modes)}, got {self.latent_prior_mode!r}"
                )
            self.fixed_uniform_sphere = (
                self.latent_prior_mode == "fixed_uniform_sphere"
            )
            self.learned_sphere_prior = (
                self.latent_prior_mode == "learned_sphere_prior"
            )
            self.sphere_task_prior = (
                self.fixed_uniform_sphere or self.learned_sphere_prior
            )
            self.learned_sphere_prior_noise_scale = float(
                self.modal_cfg.get("learned_sphere_prior_noise_scale", 0.15)
            )
            if self.learned_sphere_prior_noise_scale < 0.0:
                raise ValueError(
                    "learned_sphere_prior_noise_scale must be non-negative"
                )
            self.latent_sphere_mode = str(
                self.modal_cfg.get(
                    "latent_sphere_mode",
                    "unit" if self.sphere_task_prior else "off",
                )
            ).lower()
            self.latent_sphere_eps = float(
                self.modal_cfg.get("latent_sphere_eps", 1e-6)
            )
            valid_sphere_modes = {"off", "unit", "sqrt_dim"}
            if self.latent_sphere_mode not in valid_sphere_modes:
                raise ValueError(
                    "modal_config.latent_sphere_mode must be one of "
                    f"{sorted(valid_sphere_modes)}, got {self.latent_sphere_mode!r}"
                )
            if self.sphere_task_prior and self.latent_sphere_mode == "off":
                raise ValueError(
                    "sphere task-prior modes require latent_sphere_mode='unit' "
                    "or 'sqrt_dim'."
                )
            self.task_repr_dim = (
                int(
                    self.modal_cfg.get(
                        "fixed_sphere_task_dim",
                        self.modal_cfg.get("task_repr_dim", 128),
                    )
                )
                if self.sphere_task_prior
                else 0
            )
            if self.sphere_task_prior and self.task_repr_dim <= 0:
                raise ValueError(
                    "sphere task-prior modes require fixed_sphere_task_dim > 0"
                )
            goal_dyn_raw = self.modal_cfg.get("goal_dynamics_head_enabled", False)
            self.goal_dynamics_head_enabled = (
                goal_dyn_raw.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(goal_dyn_raw, str)
                else bool(goal_dyn_raw)
            )
            if self.goal_dynamics_head_enabled and not self.sphere_task_prior:
                raise ValueError(
                    "goal_dynamics_head_enabled is only supported with "
                    "sphere task-prior modes."
                )
            self.goal_dynamics_horizons = self._parse_int_list(
                self.modal_cfg.get("goal_dynamics_horizons", [1]), [1]
            )

            # RL finetuning options: freeze encoder/decoder but allow gradient flow
            self.freeze_encoder_decoder = self.modal_cfg.get("freeze_encoder_decoder", False)
            self.freeze_point_encoder = self.modal_cfg.get("freeze_point_encoder", False)
            self.freeze_tokenizers = self.modal_cfg.get("freeze_tokenizers", False)
            self.freeze_prior_path = self.modal_cfg.get("freeze_prior_path", False)
            self.rl_finetune_film_scale = self.modal_cfg.get("rl_finetune_film_scale", 0.1)
            self.rl_finetune_film_scope = str(
                self.modal_cfg.get("rl_finetune_film_scope", "global")
            ).lower()
            if self.rl_finetune_film_scope not in {"global", "object_target"}:
                raise ValueError("rl_finetune_film_scope must be 'global' or 'object_target'")
            # When the upstream agent uses disable_critic=True, the critic
            # forward is skipped and critic_encoder + value + modal_projection
            # never receive gradient (modal_projection only feeds critic →
            # CLAUDE.md bug 40). DDP rejects them as unused unless they are
            # frozen BEFORE the DDP wrap happens. This flag mirrors the
            # agent-side knob; set both to true together.
            self.freeze_critic_path = self.modal_cfg.get("freeze_critic_path", False)

            self._init_point_encoder()
            self._init_modal_dimensions()
            if self.rl_finetune_film_scope == "object_target":
                if not self.freeze_encoder_decoder:
                    raise ValueError("object_target FiLM scope requires freeze_encoder_decoder")
                if (self.command_dim != 13 or self.raw_point_obs_dim <= 0
                        or any((self.goal_dim, self.local_goal_dim, self.obj_trans_dim,
                                self.obj_rot_dim, self.obj_pos_dim))
                        or self.mask_dim != self.raw_point_obs_dim + self.command_dim):
                    raise ValueError("object_target FiLM scope requires the interactive command13 + point-mask layout")
            self._build_modal_encoders()
            self._build_prior_heads()
            self._build_actor_trunk()
            self._build_critic_trunk()
            if self.sphere_task_prior:
                self._freeze_sphere_task_unused_heads()

            # Apply freezing if enabled (after all modules are built, but
            # before the model is returned and DDP-wrapped).
            if self.freeze_encoder_decoder:
                self._freeze_encoder_decoder_weights()
            if self.freeze_prior_path:
                self._freeze_prior_path_weights()
            if self.freeze_critic_path:
                self._freeze_critic_path_weights()

        def forward(self, obs_dict):
            if obs_dict.get("with_vae", False):
                if obs_dict.get("with_encoder", True):
                    actor_outputs, states = self.get_action_and_vae_outputs(obs_dict)
                else:
                    actor_outputs, states = self.get_prior_action_and_vae_outputs(obs_dict)
            else:
                actor_outputs = self.act(obs_dict)
                states = obs_dict.get("rnn_states", None)

            obs = obs_dict["obs"]
            if obs_dict.get("skip_critic", False):
                value = torch.zeros(
                    obs.shape[0], 1, device=obs.device, dtype=obs.dtype
                )
            else:
                value = self.eval_critic(obs_dict)
            return actor_outputs + (value, states)

        # ------------------------------------------------------------------ #
        # Initialization helpers
        # ------------------------------------------------------------------ #
        def _init_point_encoder(self):
            cfg = self.point_cloud_cfg
            self.point_encoder_enabled = cfg.get("enabled", True)
            self.point_dim = cfg.get("point_dim", 3)
            self.num_points = cfg.get("num_points", max(1, cfg.get("raw_obs_dim", 0) // max(1, self.point_dim)))
            default_raw_dim = self.num_points * self.point_dim
            self.raw_point_obs_dim = cfg.get("raw_obs_dim", default_raw_dim)
            encoder_hidden = cfg.get("hidden_dim", 128)
            encoder_output = cfg.get("output_dim", 128)

            if self.point_encoder_enabled and self.raw_point_obs_dim > 0:
                self.point_encoder = SimplePointEncoder(
                    point_dim=self.point_dim,
                    hidden_dim=encoder_hidden,
                    output_dim=encoder_output,
                )
                self.point_token_proj = nn.Sequential(
                    nn.Linear(encoder_output, encoder_hidden),
                    nn.ReLU(),
                    nn.Linear(encoder_hidden, self.token_dim),
                )
            else:
                self.point_encoder = None
                self.point_token_proj = (
                    nn.Linear(self.raw_point_obs_dim, self.token_dim)
                    if self.raw_point_obs_dim > 0
                    else None
                )

        def _init_modal_dimensions(self):
            cfg = self.modal_cfg
            self.goal_dim = cfg.get("goal_dim", 3)
            self.command_dim = cfg.get("command_dim", 3)
            self.local_goal_dim = cfg.get("local_goal_dim", 31)
            self.obj_trans_dim = cfg.get("obj_trans_dim", 3)
            self.obj_rot_dim = cfg.get("obj_rot_dim", 6)
            self.obj_pos_dim = cfg.get("obj_pos_dim", 3)

            default_task = (
                self.obj_trans_dim
                + self.obj_rot_dim
                + self.obj_pos_dim
                + self.raw_point_obs_dim
            )

            self.task_obs_dim = cfg.get("task_obs_dim", max(0, default_task))

            default_mask = (
                self.goal_dim
                + self.local_goal_dim
                + self.obj_trans_dim
                + self.obj_rot_dim
                + self.obj_pos_dim
                + self.raw_point_obs_dim
                + self.command_dim
            )
            self.mask_dim = cfg.get("mask_dim", max(0, default_mask))
            self.expected_mask_dim = max(0, default_mask)

            reserved = self.goal_dim + self.command_dim + self.task_obs_dim + self.mask_dim
            reserved += self.local_goal_dim
            self.body_dim = cfg.get("proprio_dim", self.obs_dim - reserved)
            if self.body_dim <= 0:
                self.body_dim = max(1, self.obs_dim - (self.goal_dim + self.task_obs_dim))

            consumed = (
                self.goal_dim
                + self.command_dim
                + self.local_goal_dim
                + self.body_dim
                + self.task_obs_dim
                + self.mask_dim
            )
            if consumed < self.obs_dim:
                self.body_dim += self.obs_dim - consumed

            tokens = 0
            tokens += int(self.body_dim > 0)
            tokens += int(self.command_dim > 0)
            tokens += int(self.goal_dim > 0)
            tokens += int(self.local_goal_dim > 0)
            tokens += int(self.obj_trans_dim > 0)
            tokens += int(self.obj_rot_dim > 0)
            tokens += int(self.obj_pos_dim > 0)
            tokens += int(self.raw_point_obs_dim > 0)
            tokens += int(self.mask_dim > 0)
            self.num_modal_tokens = max(1, tokens)
            self.total_tokens = self.num_modal_tokens + 1  # add context token
            self.flat_token_dim = self.num_modal_tokens * self.token_dim
            if self.debug_modal_layout:
                expected = (
                    self.goal_dim
                    + self.command_dim
                    + self.local_goal_dim
                    + self.body_dim
                    + self.task_obs_dim
                    + self.mask_dim
                )
                print(
                    "[UltraBuilder] obs_dim={} expected_total={} goal_dim={} command_dim={} local_goal_dim={} "
                    "body_dim={} task_obs_dim={} mask_dim={} raw_point_obs_dim={} obj_trans_dim={} obj_rot_dim={} "
                    "obj_pos_dim={} tokens={}".format(
                        self.obs_dim,
                        expected,
                        self.goal_dim,
                        self.command_dim,
                        self.local_goal_dim,
                        self.body_dim,
                        self.task_obs_dim,
                        self.mask_dim,
                        self.raw_point_obs_dim,
                        self.obj_trans_dim,
                        self.obj_rot_dim,
                        self.obj_pos_dim,
                        self.num_modal_tokens,
                    )
                )
                print("default_task", default_task)

        def _build_modal_encoders(self):
            hidden = max(self.token_dim, self.modal_hidden)
            activation = nn.GELU()

            self.goal_proj = (
                nn.Sequential(
                    nn.Linear(self.goal_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.goal_dim > 0
                else None
            )

            self.command_proj = (
                nn.Sequential(
                    nn.Linear(self.command_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.command_dim > 0
                else None
            )

            self.body_proj = nn.Sequential(
                nn.Linear(self.body_dim, hidden),
                activation,
                nn.Linear(hidden, self.token_dim),
            )
            self.local_goal_proj = (
                nn.Sequential(
                    nn.Linear(self.local_goal_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.local_goal_dim > 0
                else None
            )
            self.obj_trans_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_trans_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_trans_dim > 0
                else None
            )
            self.obj_rot_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_rot_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_rot_dim > 0
                else None
            )
            self.obj_pos_proj = (
                nn.Sequential(
                    nn.Linear(self.obj_pos_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.obj_pos_dim > 0
                else None
            )
            self.mask_proj = (
                nn.Sequential(
                    nn.Linear(self.mask_dim, hidden),
                    activation,
                    nn.Linear(hidden, self.token_dim),
                )
                if self.mask_dim > 0
                else None
            )

            self.context_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
            self.pos_encoding = PositionalEncoding(
                d_model=self.token_dim, max_len=self.total_tokens + 1
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.transformer_ffn,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.modal_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=self.transformer_layers
            )
            self.modal_projection = nn.Sequential(
                nn.Linear(self.flat_token_dim, self.modal_hidden),
                nn.LayerNorm(self.modal_hidden),
                activation,
                nn.Linear(self.modal_hidden, self.modal_hidden),
                activation,
            )
            self.mask_context_proj = nn.Linear(self.mask_dim, self.token_dim)

        def _build_prior_heads(self):
            hidden = max(128, self.token_dim)
            self.task_repr_head = (
                nn.Sequential(
                    nn.Linear(self.token_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, self.task_repr_dim),
                    nn.LayerNorm(self.task_repr_dim),
                )
                if self.sphere_task_prior
                else None
            )
            self.prior_mu_head = nn.Sequential(
                nn.Linear(self.token_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.vae_dim),
            )
            self.prior_logvar_head = nn.Sequential(
                nn.Linear(self.token_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.vae_dim),
            )
            latent_decoder_input_dim = self.vae_dim + self.task_repr_dim
            self.latent_aux_dim = (
                self.command_dim
                + self.goal_dim
                + self.obj_trans_dim
                + self.obj_rot_dim
            )
            self.latent_decoder = nn.Sequential(
                nn.Linear(latent_decoder_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_aux_dim),
            )
            self.goal_dynamics_head = (
                nn.Sequential(
                    nn.Linear(latent_decoder_input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(
                        256,
                        self.latent_aux_dim * len(self.goal_dynamics_horizons),
                    ),
                )
                if self.goal_dynamics_head_enabled
                else None
            )

            self.encoder_input_dim = (
                self.teacher_obs_dim + self.obs_dim + self.mask_dim
            )
            self.encoder_net = nn.Sequential(
                nn.Linear(self.encoder_input_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.encoder_mu_head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.vae_dim)
            )
            self.encoder_logvar_head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.vae_dim)
            )

        def _build_actor_trunk(self):
            self.actor_hidden_dim = self.mu.in_features
            self.actor_mlp_input_dim = self._infer_mlp_input_dim(
                self.actor_mlp, self.actor_hidden_dim
            )

            self.actor_encoder_input = self.body_dim + self.vae_dim + self.task_repr_dim
            self.actor_encoder = nn.Sequential(
                nn.Linear(self.actor_encoder_input, self.actor_mlp_input_dim),
                nn.ReLU(),
                nn.Linear(self.actor_mlp_input_dim, self.actor_mlp_input_dim),
                nn.ReLU(),
            )
            self.h_norm = nn.LayerNorm(self.actor_mlp_input_dim)
            self.z_norm = nn.LayerNorm(self.vae_dim, elementwise_affine=False)
            self.film = nn.Linear(self.vae_dim, 2 * self.actor_mlp_input_dim)
            self.local_goal_skip_scale = self.modal_cfg.get("local_goal_skip_scale", 0.1)
            if self.local_goal_dim > 0:
                self.local_goal_skip = nn.Linear(self.local_goal_dim, self.actor_mlp_input_dim)
                nn.init.zeros_(self.local_goal_skip.weight)
                nn.init.zeros_(self.local_goal_skip.bias)
                self.local_goal_pred_head = nn.Sequential(
                    nn.Linear(self.actor_encoder_input, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.local_goal_dim),
                )
            else:
                self.local_goal_skip = None
                self.local_goal_pred_head = None

        def _build_critic_trunk(self):
            self.critic_hidden_dim = self.value.in_features
            self.critic_encoder = nn.Sequential(
                nn.Linear(self.modal_hidden, self.critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim),
                nn.ReLU(),
            )
            if self.critic_separate_backbone:
                critic_input_dim = self.teacher_obs_dim + self.obs_dim + self.mask_dim
                units = [self.critic_mlp_hidden] * max(1, int(self.critic_mlp_layers))
                self.critic_mlp = self._build_mlp(
                    input_size=critic_input_dim,
                    units=units,
                    activation=self.activation,
                    dense_func=torch.nn.Linear,
                    norm_func_name=self.normalization,
                    d2rl=False,
                    norm_only_first_layer=self.norm_only_first_layer,
                )
                self.critic_value = nn.Linear(units[-1], 1)

        def _freeze_critic_path_weights(self):
            """Freeze critic_encoder + value + modal_projection.

            With `disable_critic=True` (agent-side), the critic forward is
            skipped and these modules never receive gradient. modal_projection
            is included because its only consumer is critic_encoder
            (CLAUDE.md bug 40 — when critic_separate_backbone=False, modal
            output `fused` feeds critic_encoder; with critic skipped, fused
            goes nowhere).

            Must run BEFORE DDP wrapping. The agent's `disable_critic` flag
            should be paired with `modal_config.freeze_critic_path=true`.
            """
            critic_modules = []
            for name in ('critic_encoder', 'value', 'modal_projection'):
                mod = getattr(self, name, None)
                if mod is not None:
                    critic_modules.append((name, mod))
            n_frozen = 0
            for _, mod in critic_modules:
                if isinstance(mod, nn.Parameter):
                    mod.requires_grad = False
                    n_frozen += 1
                else:
                    for p in mod.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            n_frozen += 1
            print(f"[UltraBuilder] Critic path frozen "
                  f"({n_frozen} params across "
                  f"{[n for n, _ in critic_modules]}). DDP can run with "
                  f"find_unused_parameters=False.")

        def _freeze_encoder_decoder_weights(self):
            """Freeze encoder and decoder weights for RL finetuning.

            Freezes:
            - Encoder network (encoder_net, encoder_mu_head, encoder_logvar_head)
            - Decoder/actor trunk (actor_encoder, actor_mlp, mu, sigma, film, h_norm, z_norm)
            - Local goal prediction head and skip connection
            - Latent decoder

            Does NOT freeze:
            - Prior network (modal encoders, transformer, prior heads) - these are optimized
            - RL finetune FiLM layer - this modulates the frozen decoder
            - Critic network - still trainable for value estimation
            """
            # Freeze encoder
            encoder_modules = [
                self.encoder_net,
                self.encoder_mu_head,
                self.encoder_logvar_head,
            ]
            for module in encoder_modules:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False

            # Freeze decoder/actor trunk
            decoder_modules = [
                self.actor_encoder,
                self.actor_mlp,
                self.mu,
                self.film,
                self.h_norm,
                self.z_norm,
                self.latent_decoder,
            ]
            for module in decoder_modules:
                if module is not None:
                    if isinstance(module, nn.Parameter):
                        module.requires_grad = False
                    else:
                        for param in module.parameters():
                            param.requires_grad = False

            # Handle sigma separately - it can be nn.Parameter (fixed_sigma) or nn.Module
            if self.sigma is not None:
                if isinstance(self.sigma, nn.Parameter):
                    self.sigma.requires_grad = False
                else:
                    for param in self.sigma.parameters():
                        param.requires_grad = False

            # Freeze local goal modules if they exist
            if self.local_goal_skip is not None:
                for param in self.local_goal_skip.parameters():
                    param.requires_grad = False
            if self.local_goal_pred_head is not None:
                for param in self.local_goal_pred_head.parameters():
                    param.requires_grad = False

            # Freeze point encoder if option is enabled
            if self.freeze_point_encoder:
                point_modules = [self.point_encoder, self.point_token_proj]
                for module in point_modules:
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False

            # Freeze modal tokenizers if option is enabled
            if self.freeze_tokenizers:
                tokenizer_modules = [
                    self.goal_proj,
                    self.command_proj,
                    self.body_proj,
                    self.local_goal_proj,
                    self.obj_trans_proj,
                    self.obj_rot_proj,
                    self.obj_pos_proj,
                    self.mask_proj,
                ]
                for module in tokenizer_modules:
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False

            # Build RL finetune FiLM layer that modulates the frozen decoder
            # This allows the prior to influence the decoder output without changing decoder weights
            self.rl_finetune_film = nn.Linear(self.vae_dim, 2 * self.actor_mlp_input_dim)
            nn.init.zeros_(self.rl_finetune_film.weight)
            nn.init.zeros_(self.rl_finetune_film.bias)

            print("[UltraBuilder] Encoder and decoder frozen for RL finetuning")
            print(f"  - Encoder modules frozen: encoder_net, encoder_mu_head, encoder_logvar_head")
            print(f"  - Decoder modules frozen: actor_encoder, actor_mlp, mu, sigma, film, h_norm, z_norm, latent_decoder")
            if self.freeze_point_encoder:
                print(f"  - Point encoder frozen: point_encoder, point_token_proj")
            if self.freeze_tokenizers:
                print(f"  - Tokenizers frozen: goal_proj, command_proj, body_proj, local_goal_proj, obj_*_proj, mask_proj")
            print(f"  - RL finetune FiLM layer added (trainable)")

        def _freeze_fixed_uniform_unused_heads(self):
            """Freeze learned Gaussian heads that fixed-sphere mode never uses."""
            unused_modules = [
                ("prior_mu_head", self.prior_mu_head),
                ("prior_logvar_head", self.prior_logvar_head),
                ("encoder_logvar_head", self.encoder_logvar_head),
            ]
            n_frozen = 0
            for _, module in unused_modules:
                if module is None:
                    continue
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        n_frozen += 1
            print(
                "[UltraBuilder] Fixed uniform sphere mode enabled; "
                f"frozen unused Gaussian heads ({n_frozen} params)."
            )

        def _freeze_sphere_task_unused_heads(self):
            """Freeze Gaussian variance heads unused by sphere task-prior modes."""
            unused_modules = [
                ("prior_logvar_head", self.prior_logvar_head),
                ("encoder_logvar_head", self.encoder_logvar_head),
            ]
            if self.fixed_uniform_sphere:
                unused_modules.append(("prior_mu_head", self.prior_mu_head))
            n_frozen = 0
            frozen_names = []
            for name, module in unused_modules:
                if module is None:
                    continue
                module_frozen = False
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        n_frozen += 1
                        module_frozen = True
                if module_frozen:
                    frozen_names.append(name)
            print(
                "[UltraBuilder] Sphere task-prior mode enabled; "
                f"frozen unused Gaussian heads ({n_frozen} params across "
                f"{frozen_names})."
            )

        def _freeze_prior_path_weights(self):
            """Freeze the pretrained goal prior while leaving RL adapters usable."""
            prior_modules = [
                ("goal_proj", self.goal_proj),
                ("command_proj", self.command_proj),
                ("body_proj", self.body_proj),
                ("local_goal_proj", self.local_goal_proj),
                ("obj_trans_proj", self.obj_trans_proj),
                ("obj_rot_proj", self.obj_rot_proj),
                ("obj_pos_proj", self.obj_pos_proj),
                ("point_encoder", self.point_encoder),
                ("point_token_proj", self.point_token_proj),
                ("mask_proj", self.mask_proj),
                ("mask_context_proj", self.mask_context_proj),
                ("modal_transformer", self.modal_transformer),
                ("modal_projection", self.modal_projection),
                ("task_repr_head", self.task_repr_head),
                ("prior_mu_head", self.prior_mu_head),
                ("prior_logvar_head", self.prior_logvar_head),
                ("context_token", self.context_token),
            ]
            n_frozen = 0
            frozen_names = []
            for name, module in prior_modules:
                if module is None:
                    continue
                if isinstance(module, nn.Parameter):
                    if module.requires_grad:
                        module.requires_grad = False
                        n_frozen += 1
                    frozen_names.append(name)
                    continue
                module_frozen = False
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        n_frozen += 1
                        module_frozen = True
                if module_frozen:
                    frozen_names.append(name)
            print(
                "[UltraBuilder] Prior path frozen for adapter-only GoalRL "
                f"({n_frozen} params across {frozen_names})"
            )

        def get_rl_finetune_trainable_params(self):
            """Get parameters that are trainable during RL finetuning.

            Returns parameters for:
            - Prior network (modal encoders, transformer, prior heads)
            - RL finetune FiLM layer
            - Critic network
            - Point encoder (unless freeze_point_encoder is True)
            - Tokenizers (unless freeze_tokenizers is True)

            Use this to create an optimizer that only updates these parameters.
            """
            trainable_params = []

            # Prior network: always trainable components
            prior_modules = [
                self.modal_transformer,
                self.modal_projection,
                self.mask_context_proj,
            ]
            if self.sphere_task_prior:
                prior_modules.append(self.task_repr_head)
                if self.learned_sphere_prior:
                    prior_modules.append(self.prior_mu_head)
                if self.goal_dynamics_head is not None:
                    prior_modules.append(self.goal_dynamics_head)
            else:
                prior_modules.extend([self.prior_mu_head, self.prior_logvar_head])

            # Only include tokenizers if not frozen
            if not self.freeze_tokenizers:
                prior_modules.extend([
                    self.goal_proj,
                    self.command_proj,
                    self.body_proj,
                    self.local_goal_proj,
                    self.obj_trans_proj,
                    self.obj_rot_proj,
                    self.obj_pos_proj,
                    self.mask_proj,
                ])

            # Only include point encoder if not frozen
            if not self.freeze_point_encoder:
                prior_modules.extend([self.point_encoder, self.point_token_proj])

            for module in prior_modules:
                if module is not None:
                    trainable_params.extend(module.parameters())

            # Context token
            trainable_params.append(self.context_token)

            # RL finetune FiLM layer
            if hasattr(self, 'rl_finetune_film'):
                trainable_params.extend(self.rl_finetune_film.parameters())

            # Critic network (always trainable)
            critic_modules = [self.critic_encoder, self.value]
            if self.critic_separate_backbone:
                critic_modules.extend([self.critic_mlp, self.critic_value])
            for module in critic_modules:
                if module is not None:
                    trainable_params.extend(module.parameters())

            return trainable_params

        def _build_critic_input(self, obs, teacher_obs=None):
            device = obs.device
            if teacher_obs is not None and teacher_obs.device != device:
                teacher_obs = teacher_obs.to(device)
            split = self._split_modalities(obs)
            mask = split.get("mask") if split is not None else None
            if mask is None:
                if self.mask_dim > 0:
                    raise ValueError(
                        f"Critic mask missing: expected mask_dim={self.mask_dim} but got None."
                    )
            else:
                if mask.device != device:
                    mask = mask.to(device)
                if mask.shape[1] != self.mask_dim:
                    raise ValueError(
                        f"Critic mask dim mismatch: expected {self.mask_dim}, got {mask.shape[1]}."
                    )
            if teacher_obs is None:
                if self.teacher_obs_dim > 0:
                    raise ValueError(
                        f"Critic teacher_obs missing: expected teacher_obs_dim={self.teacher_obs_dim}."
                    )
            else:
                if teacher_obs.shape[1] != self.teacher_obs_dim:
                    raise ValueError(
                        f"Critic teacher_obs dim mismatch: expected {self.teacher_obs_dim}, got {teacher_obs.shape[1]}."
                    )
            return torch.cat([teacher_obs, obs, mask], dim=-1)

        def _infer_mlp_input_dim(self, module, default_dim):
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            return default_dim

        # ------------------------------------------------------------------ #
        # Observation processing
        # ------------------------------------------------------------------ #
        def _split_modalities(self, obs):
            cursor = 0
            goal = (
                obs[:, cursor : cursor + self.goal_dim] if self.goal_dim > 0 else None
            )
            cursor += self.goal_dim

            command = (
                obs[:, cursor : cursor + self.command_dim]
                if self.command_dim > 0
                else None
            )
            cursor += self.command_dim

            local_goal = (
                obs[:, cursor : cursor + self.local_goal_dim]
                if self.local_goal_dim > 0
                else None
            )
            cursor += self.local_goal_dim

            body = obs[:, cursor : cursor + self.body_dim]
            cursor += self.body_dim

            task = obs[:, cursor : cursor + self.task_obs_dim]
            cursor += self.task_obs_dim

            mask = (
                obs[:, cursor : cursor + self.mask_dim] if self.mask_dim > 0 else None
            )
            cursor += self.mask_dim

            if cursor < obs.shape[1]:
                # Safety: append any leftover dims to body representation.
                raise RuntimeError("Leftover observation dimensions detected.")
                remainder = obs[:, cursor:]
                body = torch.cat([body, remainder], dim=-1)

            obj_trans = (
                task[:, : self.obj_trans_dim] if self.obj_trans_dim > 0 else None
            )
            obj_rot = (
                task[:, self.obj_trans_dim : self.obj_trans_dim + self.obj_rot_dim]
                if self.obj_rot_dim > 0
                else None
            )
            obj_pos = (
                task[:,
                    self.obj_trans_dim
                    + self.obj_rot_dim : self.obj_trans_dim
                    + self.obj_rot_dim
                    + self.obj_pos_dim
                ]
                if self.obj_pos_dim > 0
                else None
            )
            obj_points = (
                task[:,
                    self.obj_trans_dim
                    + self.obj_rot_dim
                    + self.obj_pos_dim :
                ]
                if self.raw_point_obs_dim > 0
                else None
            )

            return {
                "goal": goal,
                "command": command,
                "local_goal": local_goal,
                "body": body,
                "obj_trans": obj_trans,
                "obj_rot": obj_rot,
                "obj_pos": obj_pos,
                "obj_points": obj_points,
                "mask": mask,
            }

        def _split_mask_segments(self, mask):
            segments = {}
            if mask is None or mask.numel() == 0:
                return segments
            cursor = 0

            def take(length):
                nonlocal cursor
                if length <= 0 or cursor >= mask.shape[1]:
                    return None
                end = cursor + length
                if end > mask.shape[1]:
                    end = mask.shape[1]
                segment = mask[:, cursor:end]
                cursor = end
                return segment

            segments["goal"] = take(self.goal_dim)
            segments["local_goal"] = take(self.local_goal_dim)
            segments["obj_trans"] = take(self.obj_trans_dim)
            segments["obj_rot"] = take(self.obj_rot_dim)
            segments["obj_pos"] = take(self.obj_pos_dim)
            segments["obj_points"] = take(self.raw_point_obs_dim)
            segments["command"] = take(self.command_dim)
            return segments

        def _tokens_from_split(self, split, mask_segments=None, apply_mask=False):
            tokens = []
            mask_segments = mask_segments or {}

            def gate(name, tensor):
                if not apply_mask or tensor is None:
                    return tensor
                mask = mask_segments.get(name)
                if mask is None:
                    return tensor
                return tensor * mask

            def route(token, name):
                if not apply_mask or not self.mask_routing or token is None:
                    return token
                mask = mask_segments.get(name)
                if mask is None:
                    return token
                scale = mask.mean(dim=-1, keepdim=True)
                return token * scale

            body_token = self.body_proj(split["body"])
            tokens.append(body_token.unsqueeze(1))

            if self.command_proj is not None and split["command"] is not None:
                command_feat = gate("command", split["command"])
                command_token = self.command_proj(command_feat)
                tokens.append(route(command_token, "command").unsqueeze(1))

            if self.goal_proj is not None and split["goal"] is not None:
                goal_feat = gate("goal", split["goal"])
                goal_token = self.goal_proj(goal_feat)
                tokens.append(route(goal_token, "goal").unsqueeze(1))

            if self.local_goal_proj is not None and split["local_goal"] is not None:
                local_feat = gate("local_goal", split["local_goal"])
                local_token = self.local_goal_proj(local_feat)
                tokens.append(route(local_token, "local_goal").unsqueeze(1))

            if self.obj_trans_proj is not None and split["obj_trans"] is not None:
                trans_feat = gate("obj_trans", split["obj_trans"])
                trans_token = self.obj_trans_proj(trans_feat)
                tokens.append(route(trans_token, "obj_trans").unsqueeze(1))

            if self.obj_rot_proj is not None and split["obj_rot"] is not None:
                rot_feat = gate("obj_rot", split["obj_rot"])
                rot_token = self.obj_rot_proj(rot_feat)
                tokens.append(route(rot_token, "obj_rot").unsqueeze(1))

            if self.obj_pos_proj is not None and split["obj_pos"] is not None:
                pos_feat = gate("obj_pos", split["obj_pos"])
                pos_token = self.obj_pos_proj(pos_feat)
                tokens.append(route(pos_token, "obj_pos").unsqueeze(1))

            if self.raw_point_obs_dim > 0 and split["obj_points"] is not None:
                point_input = gate("obj_points", split["obj_points"])
                if self.point_encoder is not None:
                    point_features = self.point_encoder(point_input)
                else:
                    point_features = point_input

                point_token = (
                    self.point_token_proj(point_features)
                    if self.point_token_proj is not None
                    else point_features
                )
                tokens.append(route(point_token, "obj_points").unsqueeze(1))

            if self.mask_proj is not None and split["mask"] is not None:
                tokens.append(self.mask_proj(split["mask"]).unsqueeze(1))

            return torch.cat(tokens, dim=1)

        def _encode_modalities(self, obs, apply_mask=False):
            split = self._split_modalities(obs)
            mask_segments = self._split_mask_segments(split["mask"])
            tokens = self._tokens_from_split(split, mask_segments, apply_mask=apply_mask)
            batch = obs.shape[0]
            context = self.context_token.expand(batch, -1, -1)
            if apply_mask:
                if split["mask"] is None:
                    mask_vec = torch.zeros(batch, self.mask_dim, device=obs.device, dtype=obs.dtype)
                else:
                    pad = max(0, self.mask_dim - split["mask"].shape[1])
                    mask_vec = split["mask"]
                    if pad > 0:
                        mask_vec = torch.cat(
                            [mask_vec, torch.zeros(batch, pad, device=obs.device, dtype=obs.dtype)],
                            dim=-1,
                        )
                    elif pad < 0:
                        mask_vec = mask_vec[:, : self.mask_dim]
                context = context + self.mask_context_proj(mask_vec).unsqueeze(1)
            seq = torch.cat([context, tokens], dim=1)
            seq = self.pos_encoding(seq)
            # PPO must replay the same distribution before its first update.
            # This PyTorch build's fused inference encoder differs measurably
            # from its autograd path even with dropout=0. Use one path for
            # rollout, training and deployment without changing model weights.
            fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                encoded = self.modal_transformer(seq)
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath)
            context_vec = encoded[:, 0]
            modal_tokens = encoded[:, 1:]
            flat_tokens = modal_tokens.reshape(batch, -1)
            fused = self.modal_projection(flat_tokens)
            return {"context": context_vec, "tokens": modal_tokens, "fused": fused}

        # ------------------------------------------------------------------ #
        # VAE components
        # ------------------------------------------------------------------ #
        def _zero_latent_stats(self, obs, task_repr=None):
            batch = obs.shape[0]
            zero = obs.new_zeros(batch, self.vae_dim)
            stats = {"mu": zero, "logvar": zero}
            if task_repr is not None:
                stats["task_repr"] = task_repr
            return stats

        def _latent_decoder_input(self, z, task_repr=None):
            if self.sphere_task_prior:
                if task_repr is None:
                    raise ValueError(
                        "sphere task-prior latent decoder requires task_repr"
                    )
                return torch.cat([task_repr, z], dim=-1)
            return z

        def _goal_dynamics_pred(self, z, task_repr=None):
            if self.goal_dynamics_head is None:
                return None
            return self.goal_dynamics_head(
                self._latent_decoder_input(z, task_repr)
            )

        def _prior(self, input_dict):
            encoded = self._encode_modalities(input_dict["obs"], apply_mask=True)
            context = encoded["context"]
            if self.sphere_task_prior:
                task_repr = self.task_repr_head(context)
                if self.fixed_uniform_sphere:
                    return self._zero_latent_stats(input_dict["obs"], task_repr)
                mu = self.prior_mu_head(context)
                logvar = torch.zeros_like(mu)
                return {"mu": mu, "logvar": logvar, "task_repr": task_repr}
            mu = self.prior_mu_head(context)
            logvar = self.prior_logvar_head(context)
            return {"mu": mu, "logvar": logvar}

        def _encoder(self, input_dict):
            teacher_obs = input_dict["teacher_obs"]
            student_obs = input_dict["obs"]
            split = self._split_modalities(student_obs)
            if split["mask"] is None or split["mask"].shape[1] == 0:
                mask = torch.zeros(
                    student_obs.shape[0],
                    self.mask_dim,
                    device=student_obs.device,
                    dtype=student_obs.dtype,
                )
            else:
                pad = self.mask_dim - split["mask"].shape[1]
                mask = split["mask"]
                if pad > 0:
                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros(
                                student_obs.shape[0],
                                pad,
                                device=student_obs.device,
                                dtype=student_obs.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                elif pad < 0:
                    mask = mask[:, : self.mask_dim]
            encoder_input = torch.cat([teacher_obs, student_obs, mask], dim=-1)
            embedding = self.encoder_net(encoder_input)
            mu = self.encoder_mu_head(embedding)
            if self.sphere_task_prior:
                logvar = torch.zeros_like(mu)
            else:
                logvar = self.encoder_logvar_head(embedding)
            return {"mu": mu, "logvar": logvar}

        def _trunk(self, input_dict):
            z = input_dict["vae_latent"]
            obs = input_dict["obs"]
            split = self._split_modalities(obs)
            body_obs = split["body"]
            actor_inputs = [body_obs]
            if self.sphere_task_prior:
                task_repr = input_dict.get("task_repr")
                if task_repr is None:
                    task_repr = self._prior({"obs": obs})["task_repr"]
                    input_dict["task_repr"] = task_repr
                actor_inputs.append(task_repr)
            actor_inputs.append(z)
            actor_in = torch.cat(actor_inputs, dim=-1)
            h = self.actor_encoder(actor_in)
            h = self.h_norm(h)
            local_goal_pred = None
            if self.local_goal_pred_head is not None:
                local_goal_pred = self.local_goal_pred_head(actor_in)
                input_dict["local_goal_pred"] = local_goal_pred

            if self.local_goal_skip is not None and split["local_goal"] is not None:
                local_goal = split["local_goal"]
                mask_scale = None
                if split.get("mask") is not None:
                    mask_segments = self._split_mask_segments(split["mask"])
                    mask = mask_segments.get("local_goal")
                    if mask is not None and mask.numel() > 0:
                        mask_scale = mask.mean(dim=-1, keepdim=True)
                if mask_scale is not None and local_goal_pred is not None:
                    keep = (mask_scale > 0.5)
                    local_goal = torch.where(keep, local_goal, local_goal_pred)
                skip = self.local_goal_skip(local_goal)
                h = h + self.local_goal_skip_scale * skip

            z_norm = self.z_norm(z)
            gamma, beta = self.film(z_norm).chunk(2, dim=-1)
            gamma = 1.0 + self.film_scale * torch.tanh(gamma)
            beta = self.film_scale * torch.tanh(beta)
            h = gamma * h + beta

            # Apply RL finetune FiLM layer if encoder/decoder are frozen
            # This allows the prior network (which is trainable) to modulate
            # the frozen decoder through an additional FiLM transformation
            if self.freeze_encoder_decoder and hasattr(self, 'rl_finetune_film'):
                rl_gamma, rl_beta = self.rl_finetune_film(z_norm).chunk(2, dim=-1)
                rl_gamma = 1.0 + self.rl_finetune_film_scale * torch.tanh(rl_gamma)
                rl_beta = self.rl_finetune_film_scale * torch.tanh(rl_beta)
                adapted_h = rl_gamma * h + rl_beta
                if self.rl_finetune_film_scope == "object_target":
                    masks = self._split_mask_segments(split["mask"])
                    active = (masks["obj_points"] > 0.5).all(dim=-1, keepdim=True)
                    active = active & (masks["command"][:, 9:12] > 0.5).all(dim=-1, keepdim=True)
                    # Preserve the original hidden state exactly when this HOI
                    # target is absent, including in mixed locomotion batches.
                    h = torch.where(active, adapted_h, h)
                else:
                    h = adapted_h

            a_h = self.actor_mlp(h)

            mu = self.mu_act(self.mu(a_h))
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_h))

            # NaN-guard against rare actor-NaN events that crash multi-hour
            # DDP training via `Normal(mu, sigma)` in
            # ModelA2CContinuousLogStd.forward (CLAUDE.md bug 21).
            # IMPORTANT: in this codebase, what's named `sigma` here is
            # actually LOGSTD — the yaml uses `sigma_activation: None` +
            # `sigma_init.val: -2.9`, so `self.sigma` is a Parameter holding
            # logstd, and the model wrapper does `sigma_actual = exp(logstd)`
            # at models.py:226. So:
            #   * Replace NaN/Inf in mu with 0 (neutral action target)
            #   * Replace NaN/Inf in logstd with the init value (-2.9),
            #     which gives exp(-2.9) ≈ 0.055 — the policy's training-time
            #     exploration scale. DO NOT clamp logstd to a positive floor
            #     (that would force exp(>=1e-6) ≈ 1.0 → ~18× sigma → trashes
            #     the policy's learned action distribution).
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
            sigma = torch.nan_to_num(sigma, nan=-2.9, posinf=0.0, neginf=-10.0)

            return mu, sigma

        def forward_deploy(self, obs, vae_noise):
            """Deploy-only forward path: student obs + sampled VAE noise -> action mu.

            Path: prior(obs) -> (mu_prior, logvar_prior)
                  z = mu_prior + exp(0.5 * logvar_prior) * vae_noise
                  trunk(obs, z) -> action mu (sigma is dropped; not used at deploy)

            Skips: encoder, critic, latent_decoder, local_goal_pred.
            Designed for ONNX export — keeps the graph minimal and side-effect-free.

            Args:
                obs:       (B, obs_dim) student observation, e.g. (1, 1422)
                vae_noise: (B, vae_dim) standard-normal noise, e.g. (1, 64).
                           Pass zeros for deterministic mode.
            Returns:
                action_mu: (B, action_dim) e.g. (1, 29)
            """
            if self.sphere_task_prior:
                prior_out = self._prior({"obs": obs})
                if self.fixed_uniform_sphere:
                    z = self._project_latent(vae_noise, zero_fallback=True)
                else:
                    z = self._learned_sphere_prior_latent(prior_out, vae_noise)
                action_mu, _ = self._trunk(
                    {
                        "obs": obs,
                        "vae_latent": z,
                        "task_repr": prior_out["task_repr"],
                    }
                )
                return action_mu
            prior_out = self._prior({"obs": obs})
            z = prior_out["mu"] + torch.exp(0.5 * prior_out["logvar"]) * vae_noise
            action_mu, _ = self._trunk({"obs": obs, "vae_latent": z})
            return action_mu

        def eval_critic(self, obs):
            if isinstance(obs, dict):
                teacher_obs = obs.get("teacher_obs")
                obs = obs.get("obs")
            else:
                teacher_obs = None
            if self.critic_separate_backbone:
                critic_in = self._build_critic_input(obs, teacher_obs)
                if critic_in.device != next(self.critic_mlp.parameters()).device:
                    critic_in = critic_in.to(next(self.critic_mlp.parameters()).device)
                value_features = self.critic_mlp(critic_in)
                value = self.value_act(self.critic_value(value_features))
                return value
            modal_encoding = self._encode_modalities(obs, apply_mask=False)
            value_features = self.critic_encoder(modal_encoding["fused"])
            value = self.value_act(self.value(value_features))
            return value

        def reparameterization(self, mean, std, vae_noise):
            return mean + std * vae_noise

        def _project_latent(self, z, zero_fallback=False):
            z = torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                -20.0, 20.0
            )
            if self.latent_sphere_mode == "off":
                return z
            radius = 1.0
            if self.latent_sphere_mode == "sqrt_dim":
                radius = float(np.sqrt(max(1, self.vae_dim)))
            norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
            projected = z / norm.clamp_min(self.latent_sphere_eps) * radius
            if not zero_fallback:
                return projected
            fallback = torch.zeros_like(z)
            fallback[..., 0] = radius
            return torch.where(norm > self.latent_sphere_eps, projected, fallback)

        def _learned_sphere_prior_latent(self, prior_out, vae_noise):
            """Sample/project around the learned prior direction on the sphere."""
            z_raw = prior_out["mu"]
            if self.learned_sphere_prior_noise_scale > 0.0:
                z_raw = z_raw + self.learned_sphere_prior_noise_scale * vae_noise
            return self._project_latent(z_raw, zero_fallback=True)

        def get_latent_override_action_and_vae_outputs(self, input_dict):
            z = self._project_latent(
                input_dict["vae_latent_override"],
                zero_fallback=self.sphere_task_prior,
            )
            input_dict["vae_latent"] = z
            task_repr = input_dict.get("task_repr")
            if self.sphere_task_prior and task_repr is None:
                task_repr = self._prior(input_dict)["task_repr"]
                input_dict["task_repr"] = task_repr
            action = self._trunk(input_dict)
            aux_pred = self.latent_decoder(self._latent_decoder_input(z, task_repr))
            goal_dynamics_pred = self._goal_dynamics_pred(z, task_repr)
            zero_stats = self._zero_latent_stats(input_dict["obs"], task_repr)
            rnn_states = {
                "prior_out": zero_stats,
                "encoder_out": zero_stats,
                "aux_pred": aux_pred,
                "goal_dynamics_pred": goal_dynamics_pred,
                "local_goal_pred": input_dict.get("local_goal_pred"),
                "vae_latent": z,
            }
            return action, rnn_states

        def act(self, input_dict):
            prior_out = self._prior(input_dict)
            if self.sphere_task_prior:
                task_repr = prior_out["task_repr"]
                input_dict["task_repr"] = task_repr
                if input_dict.get("with_encoder", False):
                    encoder_out = self._encoder(input_dict)
                    z_raw = encoder_out["mu"]
                    z = self._project_latent(z_raw, zero_fallback=True)
                else:
                    if "vae_noise" not in input_dict:
                        input_dict["vae_noise"] = torch.randn_like(prior_out["mu"])
                    if self.fixed_uniform_sphere:
                        z = self._project_latent(
                            input_dict["vae_noise"], zero_fallback=True
                        )
                    else:
                        z = self._learned_sphere_prior_latent(
                            prior_out, input_dict["vae_noise"]
                        )
                input_dict["vae_latent"] = z
                action = self._trunk(input_dict)
                return action

            if input_dict.get("with_encoder", False):
                encoder_out = self._encoder(input_dict)
                mu = prior_out["mu"] + encoder_out["mu"]
                logvar = encoder_out["logvar"]
            else:
                mu = prior_out["mu"]
                logvar = prior_out["logvar"]

            z = self.reparameterization(
                mu,
                torch.exp(0.5 * logvar),
                input_dict["vae_noise"],
            )
            z = self._project_latent(z)
            input_dict["vae_latent"] = z
            action = self._trunk(input_dict)
            return action

        def get_action_and_vae_outputs(self, input_dict):
            if "vae_latent_override" in input_dict:
                return self.get_latent_override_action_and_vae_outputs(input_dict)

            prior_out = self._prior(input_dict)
            encoder_out = self._encoder(input_dict)
            if self.sphere_task_prior:
                task_repr = prior_out["task_repr"]
                z = self._project_latent(encoder_out["mu"], zero_fallback=True)
                input_dict["task_repr"] = task_repr
                input_dict["vae_latent"] = z
                action = self._trunk(input_dict)
                aux_pred = self.latent_decoder(
                    self._latent_decoder_input(z, task_repr)
                )
                goal_dynamics_pred = self._goal_dynamics_pred(z, task_repr)
                local_goal_pred = input_dict.get("local_goal_pred")
                rnn_states = {
                    "prior_out": prior_out,
                    "encoder_out": encoder_out,
                    "aux_pred": aux_pred,
                    "goal_dynamics_pred": goal_dynamics_pred,
                    "local_goal_pred": local_goal_pred,
                    "vae_latent": z,
                }
                return action, rnn_states

            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = encoder_out["logvar"]

            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(mu)

            z = self.reparameterization(
                mu,
                torch.exp(0.5 * logvar),
                input_dict["vae_noise"],
            )
            z = self._project_latent(z)

            input_dict["vae_latent"] = z
            action = self._trunk(input_dict)
            aux_pred = self.latent_decoder(self._latent_decoder_input(z))
            local_goal_pred = input_dict.get("local_goal_pred")
            rnn_states = {
                "prior_out": prior_out,
                "encoder_out": encoder_out,
                "aux_pred": aux_pred,
                "local_goal_pred": local_goal_pred,
                "vae_latent": z,
            }
            return action, rnn_states

        def get_prior_action_and_vae_outputs(self, input_dict):
            if "vae_latent_override" in input_dict:
                return self.get_latent_override_action_and_vae_outputs(input_dict)

            prior_out = self._prior(input_dict)
            if self.sphere_task_prior:
                if "vae_noise" not in input_dict:
                    input_dict["vae_noise"] = torch.randn_like(prior_out["mu"])
                task_repr = prior_out["task_repr"]
                if self.fixed_uniform_sphere:
                    z = self._project_latent(
                        input_dict["vae_noise"], zero_fallback=True
                    )
                else:
                    z = self._learned_sphere_prior_latent(
                        prior_out, input_dict["vae_noise"]
                    )
                input_dict["task_repr"] = task_repr
                input_dict["vae_latent"] = z
                action = self._trunk(input_dict)
                aux_pred = self.latent_decoder(
                    self._latent_decoder_input(z, task_repr)
                )
                goal_dynamics_pred = self._goal_dynamics_pred(z, task_repr)
                encoder_out = self._zero_latent_stats(input_dict["obs"], task_repr)
                rnn_states = {
                    "prior_out": prior_out,
                    "encoder_out": encoder_out,
                    "aux_pred": aux_pred,
                    "goal_dynamics_pred": goal_dynamics_pred,
                    "local_goal_pred": input_dict.get("local_goal_pred"),
                    "vae_latent": z,
                }
                return action, rnn_states

            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(prior_out["mu"])

            z = self.reparameterization(
                prior_out["mu"],
                torch.exp(0.5 * prior_out["logvar"]),
                input_dict["vae_noise"],
            )
            z = self._project_latent(z)

            input_dict["vae_latent"] = z
            action = self._trunk(input_dict)
            aux_pred = self.latent_decoder(self._latent_decoder_input(z))
            encoder_out = {
                "mu": torch.zeros_like(prior_out["mu"]),
                "logvar": torch.zeros_like(prior_out["logvar"]),
            }
            rnn_states = {
                "prior_out": prior_out,
                "encoder_out": encoder_out,
                "aux_pred": aux_pred,
                "local_goal_pred": input_dict.get("local_goal_pred"),
                "vae_latent": z,
            }
            return action, rnn_states

    def build(self, name, **kwargs):
        point_cfg = self.params.get("point_cloud", {})
        return UltraBuilder.Network(self.params, point_cloud_cfg=point_cfg, **kwargs)
