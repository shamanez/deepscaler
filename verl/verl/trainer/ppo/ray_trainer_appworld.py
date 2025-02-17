from typing import List, Dict
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from appworld import AppWorld, Task
from jinja2 import Template
from verl.utils.tracking import Tracking
from omegaconf import OmegaConf
from verl.utils.timer import _timer
from verl.utils.advantage import compute_grpo_advantage_multi_step

PROMPT_TEMPLATE = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.
... [rest of your template] ...
"""

class AppWorldPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, max_interactions: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_interactions = max_interactions
        self.current_world = None
        self.current_history = []

    def _create_prompt_messages(self, task: Task) -> List[Dict]:
        """Creates prompt messages for the current task"""
        dictionary = {
            "supervisor": task.supervisor,
            "instruction": task.instruction
        }
        prompt = Template(PROMPT_TEMPLATE).render(dictionary)
        
        messages: List[Dict] = []
        last_start = 0
        
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
            last_end = match.span()[0]
            if len(messages) == 0 and last_end != 0:
                raise ValueError(f"Start of the prompt has no assigned role: {prompt[:last_end]}")
            if messages:
                messages[-1]["content"] = prompt[last_start:last_end]
            mesg_type = match.group(1).lower()
            messages.append({"role": mesg_type, "content": None})
            last_start = match.span()[1]
        messages[-1]["content"] = prompt[last_start:]
        return messages

    def _generate_next_code(self, last_output: str | None = None) -> str:
        """Generate next code block using the actor model"""
        if last_output is not None:
            self.current_history.append({"role": "user", "content": last_output})
        
        # Convert history to model input format
        input_text = "\n".join([msg["content"] for msg in self.current_history])
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Generate using actor model
        gen_output = self.actor_rollout_wg.generate_sequences(DataProto.from_dict(inputs))
        code = self.tokenizer.decode(gen_output.batch["responses"][0])
        
        self.current_history.append({"role": "assistant", "content": code})
        return code

    def _create_dataloader(self):
        """Initialize task IDs for training"""
        # Load task IDs
        self.task_ids = load_task_ids(self.config.data.dataset_name)
        if self.config.data.get('num_tasks'):
            self.task_ids = self.task_ids[:self.config.data.num_tasks]
        
        print(f'Number of training tasks: {len(self.task_ids)}')

        # Set total training steps based on tasks and epochs
        total_training_steps = len(self.task_ids) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # Update optimizer configs
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        """Modified training loop to handle interactive AppWorld tasks"""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True)
        )

        self.global_steps = 0

        for epoch in range(self.config.trainer.total_epochs):
            for task_id in self.task_ids:
                # Initialize AppWorld for this task
                with AppWorld(
                    task_id=task_id,
                    experiment_name=self.config.trainer.experiment_name,
                ) as world:
                    self.current_world = world
                    self.current_history = self._create_prompt_messages(world.task)
                    
                    metrics = {}
                    timing_raw = {}
                    
                    # Initialize interaction history
                    interaction_batches = []
                    terminal_rewards = []
                    output = None
                    
                    with _timer('step', timing_raw):
                        # Multiple interaction steps per episode
                        for step in range(self.max_interactions):
                            # Generate next action
                            with _timer('gen', timing_raw):
                                code = self._generate_next_code(output)
                                output = world.execute(code)
                                reward = 1.0 if world.task_completed() else 0.0
                                
                                # Store interaction data
                                step_batch = self._create_batch_from_interaction(code, output)
                                step_batch.batch['step_reward'] = torch.tensor([reward])
                                interaction_batches.append(step_batch)
                                
                                if world.task_completed():
                                    terminal_rewards.append(reward)
                                    break
                        
                        # Process all interactions
                        combined_batch = self.combine_interaction_batches(interaction_batches)
                        combined_batch.batch['terminal_rewards'] = torch.tensor(terminal_rewards)
                        
                        # Compute advantages using GRPO
                        if self.config.algorithm.adv_estimator == 'grpo':
                            advantages, returns = compute_grpo_advantage_multi_step(
                                token_level_rewards=combined_batch.batch['token_level_rewards'],
                                eos_mask=combined_batch.batch['attention_mask'],
                                index=combined_batch.non_tensor_batch['uid'],
                                terminal_rewards=combined_batch.batch['terminal_rewards']
                            )
                            combined_batch.batch['advantages'] = advantages
                            combined_batch.batch['returns'] = returns

                        # Update policy
                        actor_metrics = self.update_policy(combined_batch)
                        metrics.update(actor_metrics)

                    # Log metrics
                    logger.log(data=metrics, step=self.global_steps)
                    self.global_steps += 1

                    if self.global_steps >= self.total_training_steps:
                        return

    def execute_in_env(self, gen_output: DataProto) -> List[str]:
        """Execute generated actions in environment"""
        responses = gen_output.batch['responses']
        decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        outputs = []
        for response in decoded_responses:
            output = self.current_world.execute(response)
            outputs.append(output)
        
        return outputs

    def compute_env_reward(self, env_outputs: List[str]) -> torch.Tensor:
        """Compute rewards from environment outputs"""
        rewards = []
        for output in env_outputs:
            reward = 1.0 if self.current_world.task_completed() else 0.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

    def combine_interaction_batches(self, batches: List[DataProto]) -> DataProto:
        """Combine multiple interaction steps into single batch for training"""
        combined = DataProto.concat(batches)
        return combined

    def _update_policy(self, code: str, output: str, reward: float):
        """Perform PPO policy update"""
        # Convert code and output to model inputs
        inputs = self.tokenizer(code, return_tensors="pt")
        
        # Create batch for PPO
        batch = DataProto.from_dict({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "rewards": torch.tensor([reward])
        })

        # Rest of PPO update logic from original trainer
        # Reference lines from original RayPPOTrainer.fit(): 